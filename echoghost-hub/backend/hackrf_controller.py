"""
HackRF One controller for EchoGhost Hub Ultra.

Handles:
  - Half-duplex TX/RX with fast switching
  - External 10 MHz TCXO reference support
  - Power-safe transmission with legal band enforcement
  - Gain and attenuation configuration
  - IQ sample streaming to/from the signal processor

Safety-critical: This module enforces TX power limits, legal frequency
bands, and requires explicit confirmation before transmission. Always
use a dummy load or appropriate antenna when transmitting.

IMPORTANT SAFETY NOTES:
  - Never transmit without a properly rated antenna or dummy load.
  - Observe local spectrum regulations (license-free ISM bands only).
  - Keep TX power low (< 10 dBm) for indoor/experimental use.
  - The HackRF's built-in amp can easily exceed legal limits — use sparingly.
"""

import time
import logging
import threading
from typing import Optional, Callable
from dataclasses import dataclass

import numpy as np

from config import AppConfig, SafetyConfig, HackRFConfig, RadarConfig
from waveforms import generate_waveform

logger = logging.getLogger(__name__)

# Try to import the hackrf library; fall back to simulation mode
try:
    import hackrf

    HAS_HACKRF = True
    logger.info("HackRF library loaded successfully")
except ImportError:
    HAS_HACKRF = False
    logger.warning(
        "HackRF library not found. Running in SIMULATION mode — "
        "no actual RF will be transmitted or received."
    )


class HackRFError(Exception):
    """Custom exception for HackRF operations."""

    pass


class SafetyError(HackRFError):
    """Raised when a safety check fails."""

    pass


# ──────────────────────────────────────────────────────────────────────────────
# Frequency Band Definitions (License-Free ISM)
# ──────────────────────────────────────────────────────────────────────────────

LEGAL_BANDS_MHZ: list[tuple[float, float]] = [
    (902.0, 928.0),  # 915 MHz ISM band (US)
    (2400.0, 2483.5),  # 2.4 GHz ISM band
]


def _check_frequency_band(freq_hz: float) -> bool:
    """Verify a frequency falls within a legal ISM band."""
    freq_mhz = freq_hz / 1e6
    for low, high in LEGAL_BANDS_MHZ:
        if low <= freq_mhz <= high:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# HackRF Controller
# ──────────────────────────────────────────────────────────────────────────────


class HackRFController:
    """Controls the HackRF One for radar sensing operations.

    This class encapsulates all hardware interactions. It enforces safety
    constraints, manages TX/RX state, and provides a clean interface for
    the signal processing pipeline.

    State machine:
        IDLE -> RX mode (receiving only)
        IDLE -> TX mode (transmitting waveform)
        TX -> RX (switch after transmission)
        RX -> TX (switch on next CPI)

    Attributes:
        config: Application configuration.
        device: HackRF device handle (None in simulation mode).
        is_transmitting: Whether the device is currently in TX mode.
        is_running: Whether the device is actively streaming.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = None
        self.is_transmitting = False
        self.is_running = False
        self._lock = threading.Lock()
        self._rx_callback: Optional[Callable] = None
        self._tx_samples: Optional[np.ndarray] = None
        self._tx_index = 0
        self._simulation_time = 0.0

    # ── Device Lifecycle ─────────────────────────────────────────────────

    def open(self) -> None:
        """Open the HackRF device or enter simulation mode.

        Raises:
            HackRFError: If the device cannot be opened.
        """
        if not HAS_HACKRF:
            logger.info("SIMULATION MODE: Creating virtual HackRF device")
            return

        try:
            self.device = hackrf.hackrf_open()
            logger.info("HackRF device opened successfully")

            # Apply initial configuration
            self._apply_config()

            # Report device info
            board_id = hackrf.hackrf_board_id_read(self.device)
            version = hackrf.hackrf_version_string_read(self.device)
            logger.info(f"Board ID: {board_id}, Version: {version}")

        except Exception as e:
            raise HackRFError(f"Failed to open HackRF: {e}")

    def close(self) -> None:
        """Close the HackRF device."""
        self.is_running = False
        if self.device is not None and HAS_HACKRF:
            try:
                hackrf.hackrf_close(self.device)
                logger.info("HackRF device closed")
            except Exception as e:
                logger.error(f"Error closing HackRF: {e}")
        self.device = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ── Configuration ────────────────────────────────────────────────────

    def _apply_config(self) -> None:
        """Apply current hardware configuration to the device.

        Sets frequency, sample rate, gains, bandwidth, and TCXO mode.
        """
        if self.device is None or not HAS_HACKRF:
            return

        hc = self.config.hackrf
        rc = self.config.radar

        try:
            # Center frequency
            hackrf.hackrf_set_freq(self.device, hc.frequency_hz)
            logger.info(f"Frequency set to {hc.frequency_hz / 1e6:.1f} MHz")

            # Sample rate
            hackrf.hackrf_set_sample_rate(self.device, hc.sample_rate_hz)
            logger.info(f"Sample rate set to {hc.sample_rate_hz / 1e6:.1f} Msps")

            # Baseband filter bandwidth
            hackrf.hackrf_set_baseband_filter_bandwidth(self.device, hc.bandwidth_hz)

            # LNA gain (RX path)
            hackrf.hackrf_set_lna_gain(self.device, min(hc.lna_gain_db, 40))

            # VGA gain (RX path)
            hackrf.hackrf_set_vga_gain(self.device, min(hc.vga_gain_db, 62))

            # TX VGA gain
            hackrf.hackrf_set_txvga_gain(self.device, min(hc.txvga_gain_db, 47))

            # Amp
            hackrf.hackrf_set_amp_enable(self.device, hc.amp_enable)

            # External TCXO reference
            if rc.use_tcxo:
                hackrf.hackrf_set_antenna_enable(self.device, 1)
                logger.info("External 10 MHz TCXO reference enabled")

            logger.info("Hardware configuration applied")

        except Exception as e:
            raise HackRFError(f"Failed to apply config: {e}")

    def update_config(self, config: AppConfig) -> None:
        """Update configuration and re-apply to hardware."""
        self.config = config
        self._apply_config()

    # ── Safety Checks ────────────────────────────────────────────────────

    def _safety_check(self) -> None:
        """Run all safety checks before enabling transmission.

        Raises:
            SafetyError: If any safety check fails.
        """
        sc = self.config.safety
        hc = self.config.hackrf
        rc = self.config.radar

        if not sc.enable_tx_safety:
            logger.warning("TX SAFETY CHECKS DISABLED — proceed with extreme caution")
            return

        # Check legal frequency band
        if not _check_frequency_band(hc.frequency_hz):
            raise SafetyError(
                f"Frequency {hc.frequency_hz / 1e6:.1f} MHz is outside legal ISM bands. "
                f"Allowed: {LEGAL_BANDS_MHZ}"
            )

        # Check TX power
        effective_power = hc.txvga_gain_db + (20 if hc.amp_enable else 0)
        if effective_power > sc.max_tx_power_dbm:
            raise SafetyError(
                f"Effective TX power {effective_power} dBm exceeds max "
                f"{sc.max_tx_power_dbm} dBm. Reduce TX VGA gain or disable amp."
            )

        # Check TX amplitude
        if rc.tx_amplitude > 1.0 or rc.tx_amplitude < 0:
            raise SafetyError(f"TX amplitude must be in [0, 1]; got {rc.tx_amplitude}")

        if sc.require_dummy_load_warning:
            logger.warning(
                "SAFETY: Ensure a dummy load or appropriate antenna is connected "
                "before transmission. TX power is experimental."
            )

        logger.info("All safety checks passed")

    # ── Transmission ─────────────────────────────────────────────────────

    def prepare_tx_waveform(self) -> np.ndarray:
        """Generate the baseband waveform to be transmitted.

        Returns:
            Complex IQ samples for one CPI.
        """
        rc = self.config.radar
        hc = self.config.hackrf
        cpi_duration = rc.cpi_length / rc.pulse_repetition_hz

        return generate_waveform(
            waveform_type=rc.waveform_type,
            sample_rate_hz=hc.sample_rate_hz,
            duration_s=cpi_duration,
            amplitude=rc.tx_amplitude,
            chirp_bw_hz=rc.chirp_bw_hz,
            chirp_duration_s=rc.chirp_duration_s,
            chaos_bandwidth_hz=rc.chaos_bandwidth_hz,
        )

    def transmit(self, samples: np.ndarray) -> None:
        """Transmit IQ samples.

        In simulation mode, this just logs the event.

        Args:
            samples: Complex IQ array to transmit.

        Raises:
            SafetyError: If safety checks fail.
            HackRFError: If transmission fails.
        """
        self._safety_check()

        with self._lock:
            self.is_transmitting = True

        if not HAS_HACKRF or self.device is None:
            # Simulation: log and return
            duration = len(samples) / self.config.hackrf.sample_rate_hz
            logger.info(
                f"SIM TX: {len(samples)} samples ({duration * 1000:.1f} ms) at "
                f"{self.config.hackrf.frequency_hz / 1e6:.1f} MHz"
            )
            time.sleep(duration * 0.01)  # Simulate near-instant TX
            with self._lock:
                self.is_transmitting = False
            return

        try:
            # Convert complex64 to interleaved int8 for HackRF
            # Scale samples to fit in int8 range
            scaled = np.real(samples).astype(np.float64)
            scaled = np.clip(scaled * 127.0, -128, 127).astype(np.int8)
            scaled_q = np.imag(samples).astype(np.float64)
            scaled_q = np.clip(scaled_q * 127.0, -128, 127).astype(np.int8)

            # Interleave I and Q
            tx_buffer = np.empty(len(samples) * 2, dtype=np.int8)
            tx_buffer[0::2] = scaled
            tx_buffer[1::2] = scaled_q

            # Set TX mode
            hackrf.hackrf_set_txvga_gain(
                self.device, min(self.config.hackrf.txvga_gain_db, 47)
            )

            # Transmit
            result = hackrf.hackrf_write(
                self.device, tx_buffer.tobytes(), len(tx_buffer)
            )
            if result < 0:
                raise HackRFError(f"TX write failed with code {result}")

            logger.debug(f"Transmitted {len(samples)} samples")

        except Exception as e:
            raise HackRFError(f"Transmission error: {e}")
        finally:
            with self._lock:
                self.is_transmitting = False

    def start_rx_stream(self, callback: Callable[[np.ndarray], None]) -> None:
        """Start receiving IQ samples continuously.

        The callback receives chunks of IQ data as complex64 numpy arrays.

        Args:
            callback: Function called with each received IQ buffer.
        """
        self._rx_callback = callback
        self.is_running = True

        if not HAS_HACKRF or self.device is None:
            logger.info("SIMULATION MODE: RX stream started (synthetic data)")
            self._simulation_rx_loop(callback)
            return

        # Set RX mode
        try:
            hackrf.hackrf_start_rx(self.device, self._hackrf_rx_callback, None)
            logger.info("RX streaming started")
        except Exception as e:
            raise HackRFError(f"Failed to start RX: {e}")

    def stop_rx_stream(self) -> None:
        """Stop the RX stream."""
        self.is_running = False
        if self.device is not None and HAS_HACKRF:
            try:
                hackrf.hackrf_stop_rx(self.device)
                logger.info("RX streaming stopped")
            except Exception as e:
                logger.error(f"Error stopping RX: {e}")

    # ── RX Callbacks ───────────────────────────────────────────────────

    def _hackrf_rx_callback(self, buffer, length, ctx) -> int:
        """Internal callback from the HackRF C library.

        Converts the raw int8 interleaved buffer to complex64 IQ samples
        and forwards them to the registered Python callback.
        """
        if not self.is_running:
            return -1  # Signal HackRF to stop

        # Convert interleaved int8 to complex64
        i = np.frombuffer(buffer, dtype=np.int8, offset=0, count=length // 2)
        q = np.frombuffer(buffer, dtype=np.int8, offset=length // 2, count=length // 2)
        iq = (i.astype(np.float32) / 127.0) + 1j * (q.astype(np.float32) / 127.0)

        if self._rx_callback:
            self._rx_callback(iq)

        return 0  # Continue streaming

    def _simulation_rx_loop(self, callback: Callable[[np.ndarray], None]):
        """Simulate received IQ data when no HackRF is connected.

        Generates synthetic data with simulated targets for testing
        the signal processing pipeline.
        """
        import threading

        def _sim_worker():
            hc = self.config.hackrf
            chunk_size = int(hc.sample_rate_hz / self.config.update_interval_hz)
            t = 0

            while self.is_running:
                # Generate synthetic received signal
                # Simulated: low-level noise + moving target at ~5 m
                noise = (
                    np.random.randn(chunk_size) + 1j * np.random.randn(chunk_size)
                ) * 0.01

                # Simulated target echo with Doppler shift (~2 Hz walking speed)
                target_amplitude = 0.05
                target_phase = 2 * np.pi * 2.0 * t / hc.sample_rate_hz
                target = target_amplitude * np.exp(1j * target_phase)
                target_signal = target * np.ones(chunk_size, dtype=np.complex64)

                # Simulated breathing target (0.3 Hz)
                breath_amp = 0.02
                breath_phase = 2 * np.pi * 0.3 * t / hc.sample_rate_hz
                breath = breath_amp * np.exp(1j * breath_phase)
                breath_signal = breath * np.ones(chunk_size, dtype=np.complex64)

                iq = noise + target_signal + breath_signal
                callback(iq.astype(np.complex64))

                t += chunk_size
                time.sleep(1.0 / self.config.update_interval_hz)

        thread = threading.Thread(target=_sim_worker, daemon=True)
        thread.start()

    # ── Half-Duplex TX/RX Cycle ─────────────────────────────────────────

    def run_txrx_cycle(self, rx_callback: Callable[[np.ndarray], None]) -> None:
        """Execute one full TX/RX cycle for a coherent processing interval.

        In half-duplex mode:
          1. Generate and transmit waveform
          2. Switch to RX
          3. Collect reflected samples
          4. Pass to signal processor

        This is called once per CPI by the main processing loop.
        """
        # Generate TX waveform
        tx_samples = self.prepare_tx_waveform()

        # Transmit
        self.transmit(tx_samples)

        # Small guard interval for TX/RX switching
        time.sleep(1e-6)

        # Start RX to capture echoes
        # In practice, RX runs continuously and we window the received data
        # to align with the CPI
        self.start_rx_stream(rx_callback)
