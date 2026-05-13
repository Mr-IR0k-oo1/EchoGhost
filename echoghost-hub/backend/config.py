import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "models", "default_config.json"
)


@dataclass
class HackRFConfig:
    frequency_hz: int = 915_000_000  # Center frequency (915 MHz or 2.4 GHz)
    sample_rate_hz: float = 20e6  # 20 MSps
    lna_gain_db: int = 16  # LNA gain 0-40 dB (8 dB steps)
    vga_gain_db: int = 16  # VGA gain 0-62 dB (2 dB steps)
    amp_enable: bool = False  # Built-in amp (use with CAUTION)
    bandwidth_hz: float = 20e6  # RX/TX baseband filter BW
    txvga_gain_db: int = 10  # TX VGA gain 0-47 dB (1 dB steps)


@dataclass
class RadarConfig:
    mode: str = "active_chaotic"  # active_chaotic | passive_wifi | motion | breathing | micro_doppler | range_doppler | through_wall | gesture | artistic | security
    waveform_type: str = "chaotic"  # tone | fmcw | chaotic | noise
    pulse_repetition_hz: float = 1000.0  # PRF
    chirp_bw_hz: float = 20e6  # FMCW bandwidth
    chirp_duration_s: float = 1e-3  # Chirp length
    max_range_m: float = 30.0  # Max unambiguous range
    range_bins: int = 128  # Number of range bins
    doppler_bins: int = 64  # Number of Doppler bins
    cpi_length: int = 256  # Coherent processing interval (pulses)
    chaos_bandwidth_hz: float = 20e6  # Chaotic waveform bandwidth
    tx_amplitude: float = 0.5  # TX waveform amplitude (keep low for safety)
    use_tcxo: bool = True  # Use external 10 MHz TCXO reference


@dataclass
class ProcessingConfig:
    baseline_alpha: float = 0.01  # Exponential baseline removal factor
    breathing_low_cut: float = 0.1  # Breathing bandpass low (Hz)
    breathing_high_cut: float = 0.8  # Breathing bandpass high (Hz)
    motion_threshold: float = 0.05  # Motion detection threshold
    doppler_fft_size: int = 256  # FFT size for Doppler processing
    micro_doppler_overlap: float = 0.9  # STFT overlap ratio
    numba_enabled: bool = True  # Use Numba JIT acceleration
    enable_ml_classifier: bool = True  # Enable scikit-learn activity classifier


@dataclass
class SafetyConfig:
    max_tx_power_dbm: int = 10  # Absolute max TX power (legal limit may be lower)
    default_tx_power_dbm: int = 5  # Default safe TX power
    enable_tx_safety: bool = True  # Enforce TX power limits
    require_dummy_load_warning: bool = True
    legal_bands_mhz: list = field(default_factory=lambda: [(902, 928), (2400, 2483.5)])
    max_continuous_tx_seconds: float = 30.0
    cooldown_seconds: float = 10.0


@dataclass
class AppConfig:
    hackrf: HackRFConfig = field(default_factory=HackRFConfig)
    radar: RadarConfig = field(default_factory=RadarConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    websocket_host: str = "0.0.0.0"
    websocket_port: int = 8000
    update_interval_hz: float = 20.0  # WebSocket stream rate

    def save(self, path: str = DEFAULT_CONFIG_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str = DEFAULT_CONFIG_PATH) -> "AppConfig":
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return cls(
                hackrf=HackRFConfig(**data.get("hackrf", {})),
                radar=RadarConfig(**data.get("radar", {})),
                processing=ProcessingConfig(**data.get("processing", {})),
                safety=SafetyConfig(**data.get("safety", {})),
                websocket_host=data.get("websocket_host", "0.0.0.0"),
                websocket_port=data.get("websocket_port", 8000),
                update_interval_hz=data.get("update_interval_hz", 20.0),
            )
        return cls()

    @classmethod
    def create_default_config_file(cls, path: str = DEFAULT_CONFIG_PATH):
        cls().save(path)
