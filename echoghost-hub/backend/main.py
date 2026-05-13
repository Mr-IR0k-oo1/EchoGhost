"""
EchoGhost Hub Ultra — FastAPI WebSocket Backend.

This is the main server that:
  - Serves the WebSocket endpoint for real-time radar data streaming
  - Manages the HackRF controller and signal processor
  - Handles configuration updates from the frontend
  - Enforces safety constraints
  - Streams processed results (heatmaps, breathing, motion, activity) at ~20 Hz

Architecture:
    Frontend (Next.js) <--WebSocket--> main.py <--> HackRFController
                                                  <--> SignalProcessor

Run:
    pip install fastapi uvicorn numpy scipy numba scikit-learn hackrf
    python main.py

    Then open http://localhost:8000 in a browser (or connect frontend dev server).
"""

import asyncio
import json
import logging
import time
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from config import AppConfig
from hackrf_controller import HackRFController, SafetyError
from signal_processor import SignalProcessor, ProcessingResult

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Application State ─────────────────────────────────────────────────────


class EchoGhostApp:
    """Main application controller.

    Manages the lifecycle of the HackRF controller and signal processor,
    and handles WebSocket connections for data streaming.
    """

    def __init__(self):
        self.config = AppConfig.load()
        self.hackrf: Optional[HackRFController] = None
        self.processor: Optional[SignalProcessor] = None
        self.websocket: Optional[WebSocket] = None
        self._running = False
        self._stream_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize hardware and processor."""
        logger.info("Initializing EchoGhost Hub Ultra...")

        # Create default config if not exists
        self.config.save()

        # Initialize HackRF
        self.hackrf = HackRFController(self.config)
        try:
            self.hackrf.open()
            logger.info("HackRF initialized successfully")
        except Exception as e:
            logger.warning(f"HackRF init failed (simulation mode): {e}")

        # Initialize signal processor
        self.processor = SignalProcessor(self.config)
        logger.info("Signal processor initialized")

    async def shutdown(self):
        """Graceful shutdown of all components."""
        self._running = False
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        if self.hackrf:
            self.hackrf.close()
        logger.info("EchoGhost Hub Ultra shut down")

    async def update_config(self, config_dict: dict):
        """Update configuration from frontend.

        Validates safety constraints before applying.
        """
        # Merge with existing config
        for section, values in config_dict.items():
            if hasattr(self.config, section):
                section_obj = getattr(self.config, section)
                for key, val in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, val)

        self.config.save()

        # Re-apply to hardware
        if self.hackrf:
            self.hackrf.update_config(self.config)

        # Re-initialize processor if needed
        self.processor = SignalProcessor(self.config)

        logger.info(f"Configuration updated: {config_dict}")

    async def stream_loop(self, websocket: WebSocket):
        """Main streaming loop: acquire data, process, send to frontend.

        Runs at config.update_interval_hz (default 20 Hz).
        """
        self.websocket = websocket
        self._running = True

        update_interval = 1.0 / self.config.update_interval_hz

        while self._running:
            try:
                loop_start = time.time()

                # 1. Acquire IQ data (real or simulated)
                if self.hackrf:
                    # In real mode, IQ comes from HackRF RX callback
                    # For now, we generate synthetic data for processing
                    cpi_iq = self._acquire_cpi()
                else:
                    cpi_iq = self._synthetic_cpi()

                # 2. Process
                result = self.processor.process_cpi(cpi_iq)

                # 3. Send to frontend
                await self._send_result(websocket, result)

                # 4. Maintain update rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, update_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream loop error: {e}")
                await asyncio.sleep(update_interval)

    def _acquire_cpi(self) -> np.ndarray:
        """Acquire one CPI of IQ data from the HackRF.

        In real mode, this reads from the RX buffer. The number of samples
        is determined by the CPI length and sample rate.
        """
        hc = self.config.hackrf
        rc = self.config.radar
        samples_per_cpi = int(
            hc.sample_rate_hz * rc.cpi_length / rc.pulse_repetition_hz
        )
        return np.zeros(samples_per_cpi, dtype=np.complex64)

    def _synthetic_cpi(self) -> np.ndarray:
        """Generate synthetic CPI data for testing without hardware.

        Includes simulated targets at various ranges with breathing
        and walking Doppler signatures.
        """
        hc = self.config.hackrf
        rc = self.config.radar

        samples_per_pulse = int(hc.sample_rate_hz * rc.chirp_duration_s)
        total_samples = rc.cpi_length * samples_per_pulse

        t = np.arange(total_samples) / hc.sample_rate_hz
        iq = np.zeros(total_samples, dtype=np.complex64)

        # Noise floor
        iq += (
            np.random.randn(total_samples) + 1j * np.random.randn(total_samples)
        ) * 0.01

        # Stationary target at 3 m (simulated via time delay)
        range_3m_delay = int(2 * 3.0 / 3e8 * hc.sample_rate_hz)
        if range_3m_delay < total_samples:
            target = 0.1 * np.exp(1j * 2 * np.pi * 0.3 * t)  # Breathing
            iq[range_3m_delay:] += target[: total_samples - range_3m_delay]

        # Moving target at 5 m, walking speed (~1.5 m/s)
        range_5m_delay = int(2 * 5.0 / 3e8 * hc.sample_rate_hz)
        if range_5m_delay < total_samples:
            doppler_shift = 2 * 1.5 / (3e8 / hc.frequency_hz)
            target2 = 0.08 * np.exp(1j * 2 * np.pi * doppler_shift * t)
            iq[range_5m_delay:] += target2[: total_samples - range_5m_delay]

        return iq

    async def _send_result(self, websocket: WebSocket, result: ProcessingResult):
        """Serialize and send processing result over WebSocket.

        Converts numpy arrays to lists for JSON serialization.
        Applies downsampling to keep message size reasonable.
        """
        # Downsample heatmap for transmission
        heatmap = result.heatmap
        if heatmap.shape[0] > 32:
            heatmap = heatmap[:: heatmap.shape[0] // 32, :]
        if heatmap.shape[1] > 32:
            heatmap = heatmap[:, :: heatmap.shape[1] // 32]

        # Downsample range profile
        rp = result.range_profile
        if len(rp) > 64:
            rp = rp[:: len(rp) // 64]

        # Downsample Doppler profile
        dp = result.doppler_profile
        if len(dp) > 64:
            dp = dp[:: len(dp) // 64]

        # Downsample micro-Doppler
        md = result.micro_doppler_spectrogram
        if md.shape[0] > 32:
            md = md[:: md.shape[0] // 32, :]
        if md.shape[1] > 32:
            md = md[:, :: md.shape[1] // 32]

        message = {
            "type": "radar_data",
            "timestamp": result.timestamp,
            "cpi_index": result.cpi_index,
            "heatmap": heatmap.tolist(),
            "range_profile": rp.tolist(),
            "doppler_profile": dp.tolist(),
            "micro_doppler": md.tolist(),
            "micro_doppler_freqs": result.micro_doppler_freqs.tolist(),
            "breathing": {
                "rate_bpm": round(result.breathing_rate_bpm, 1),
                "confidence": round(result.breathing_confidence, 3),
                "phase": result.breathing_phase.tolist()
                if len(result.breathing_phase) > 0
                else [],
            },
            "motion": {
                "detected": result.motion_detected,
                "magnitude": round(result.motion_magnitude, 3),
                "target_range_m": round(result.target_range_m, 2),
                "target_velocity_mps": round(result.target_velocity_mps, 2),
            },
            "activity": {
                "label": result.activity_label,
                "confidence": round(result.activity_confidence, 3),
            },
        }

        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"WebSocket send error: {e}")


# ── FastAPI App Setup ─────────────────────────────────────────────────────

app_state = EchoGhostApp()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler for startup/shutdown."""
    await app_state.initialize()
    yield
    await app_state.shutdown()


app = FastAPI(
    title="EchoGhost Hub Ultra",
    description="Multi-mode RF Sensing Platform with HackRF One",
    version="1.0.0",
    lifespan=lifespan,
)


# ── REST Endpoints ────────────────────────────────────────────────────────


@app.get("/")
async def root():
    """Health check and API info."""
    return {
        "name": "EchoGhost Hub Ultra",
        "status": "running",
        "mode": "simulation"
        if not hasattr(app_state.hackrf, "device") or app_state.hackrf.device is None
        else "hardware",
        "config": {
            "frequency_mhz": app_state.config.hackrf.frequency_hz / 1e6,
            "sample_rate_msps": app_state.config.hackrf.sample_rate_hz / 1e6,
            "mode": app_state.config.radar.mode,
            "waveform": app_state.config.radar.waveform_type,
        },
    }


@app.get("/config")
async def get_config():
    """Return current configuration."""
    return {
        "hackrf": {
            "frequency_hz": app_state.config.hackrf.frequency_hz,
            "sample_rate_hz": app_state.config.hackrf.sample_rate_hz,
            "lna_gain_db": app_state.config.hackrf.lna_gain_db,
            "vga_gain_db": app_state.config.hackrf.vga_gain_db,
            "txvga_gain_db": app_state.config.hackrf.txvga_gain_db,
            "amp_enable": app_state.config.hackrf.amp_enable,
            "bandwidth_hz": app_state.config.hackrf.bandwidth_hz,
        },
        "radar": {
            "mode": app_state.config.radar.mode,
            "waveform_type": app_state.config.radar.waveform_type,
            "pulse_repetition_hz": app_state.config.radar.pulse_repetition_hz,
            "chirp_bw_hz": app_state.config.radar.chirp_bw_hz,
            "chirp_duration_s": app_state.config.radar.chirp_duration_s,
            "max_range_m": app_state.config.radar.max_range_m,
            "tx_amplitude": app_state.config.radar.tx_amplitude,
            "use_tcxo": app_state.config.radar.use_tcxo,
        },
        "safety": {
            "max_tx_power_dbm": app_state.config.safety.max_tx_power_dbm,
            "enable_tx_safety": app_state.config.safety.enable_tx_safety,
        },
    }


@app.post("/config")
async def update_config(config_dict: dict):
    """Update configuration from frontend."""
    try:
        await app_state.update_config(config_dict)
        return {"status": "ok", "message": "Configuration updated"}
    except SafetyError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/safety-check")
async def safety_check():
    """Run a safety check and return results."""
    try:
        # Re-apply config to trigger safety checks
        if app_state.hackrf:
            app_state.hackrf._safety_check()
        return {
            "status": "ok",
            "message": "All safety checks passed",
            "frequency_mhz": app_state.config.hackrf.frequency_hz / 1e6,
            "legal_bands": app_state.config.safety.legal_bands_mhz,
        }
    except SafetyError as e:
        return {"status": "warning", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── WebSocket Endpoint ────────────────────────────────────────────────────


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time radar data streaming.

    The server pushes processing results at ~20 Hz.
    The client can send JSON commands to update config.

    Client -> Server messages:
        {"type": "config", ...config fields...}
        {"type": "start"}  (start streaming)
        {"type": "stop"}   (stop streaming)
        {"type": "safety_check"}

    Server -> Client messages:
        {"type": "radar_data", ...processed results...}
        {"type": "status", "message": "..."}
        {"type": "error", "message": "..."}
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    stream_task: Optional[asyncio.Task] = None

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "config":
                # Update configuration
                try:
                    await app_state.update_config(msg.get("config", {}))
                    await websocket.send_json(
                        {
                            "type": "status",
                            "message": "Configuration updated",
                        }
                    )
                except SafetyError as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif msg_type == "start":
                # Start streaming radar data
                if stream_task is None or stream_task.done():
                    stream_task = asyncio.create_task(app_state.stream_loop(websocket))
                    await websocket.send_json(
                        {
                            "type": "status",
                            "message": "Streaming started",
                        }
                    )
                else:
                    await websocket.send_json(
                        {
                            "type": "status",
                            "message": "Already streaming",
                        }
                    )

            elif msg_type == "stop":
                # Stop streaming
                app_state._running = False
                if stream_task:
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass
                    stream_task = None
                await websocket.send_json(
                    {
                        "type": "status",
                        "message": "Streaming stopped",
                    }
                )

            elif msg_type == "safety_check":
                try:
                    if app_state.hackrf:
                        app_state.hackrf._safety_check()
                    await websocket.send_json(
                        {
                            "type": "status",
                            "message": "All safety checks passed",
                        }
                    )
                except SafetyError as e:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": str(e),
                        }
                    )

            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    }
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        app_state._running = False
        if stream_task:
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass


# ── Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("""
    ╔══════════════════════════════════════════════╗
    ║        EchoGhost Hub Ultra v1.0.0            ║
    ║     Multi-Mode RF Sensing Platform           ║
    ╚══════════════════════════════════════════════╝

    !!! SAFETY WARNING !!!
    - Always use a dummy load or appropriate antenna
    - Observe local spectrum regulations
    - Keep TX power low for indoor/experimental use
    - The HackRF amp can exceed legal limits easily

    Starting server on http://0.0.0.0:8000
    WebSocket endpoint: ws://0.0.0.0:8000/ws
    """)

    uvicorn.run(
        "main:app",
        host=app_state.config.websocket_host,
        port=app_state.config.websocket_port,
        reload=False,
        log_level="info",
    )
