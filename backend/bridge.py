from __future__ import annotations

import os
import sys
import threading
import time

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_BACKEND = os.path.abspath(os.path.dirname(__file__))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from echoghost_hub_ultra.config.presets import (
    AdaptiveConfig,
    DashboardConfig,
    OperatingMode,
    PassiveConfig,
    ProcessingConfig,
    RadioConfig,
    SimulationConfig,
    WaveformConfig,
    WaveformKind,
)
from echoghost_hub_ultra.radio.session import DashboardSnapshot, RFSession

from config import SensingFrame, ServerStatus, SessionConfig
from serializers import serialize_frame


class SensingBridge:
    """Async bridge wrapping RFSession. Called from FastAPI background tasks."""

    def __init__(self) -> None:
        self._session: RFSession | None = None
        self._lock = threading.Lock()
        self._latest_frame: SensingFrame | None = None
        self._frame_count = 0
        self._start_time: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._session is not None and self._session.is_running

    @property
    def latest_frame(self) -> SensingFrame | None:
        with self._lock:
            return self._latest_frame

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self, config: SessionConfig) -> str:
        with self._lock:
            if self.is_running:
                return "Session already running"

            radio = RadioConfig(
                center_frequency_hz=config.center_frequency_hz,
                sample_rate_sps=config.sample_rate_sps,
                tx_gain_db=config.tx_gain_db,
                rx_gain_db=config.rx_gain_db,
                frame_size=config.frame_size,
                backend=config.backend.value,
                mode=OperatingMode(config.mode.value),
            )
            waveform = WaveformConfig(kind=WaveformKind(config.waveform.value))
            processing = ProcessingConfig()
            dashboard = DashboardConfig(refresh_hz=30.0)
            simulation = SimulationConfig()
            passive = PassiveConfig()
            adaptive = AdaptiveConfig(enabled=config.adaptive)

            self._session = RFSession(
                radio_config=radio,
                waveform_config=waveform,
                processing_config=processing,
                dashboard_config=dashboard,
                simulation_config=simulation,
                passive_config=passive,
                adaptive_config=adaptive,
            )
            self._session.start()
            self._start_time = time.monotonic()
            self._frame_count = 0
            return "Session started"

    def stop(self) -> str:
        with self._lock:
            if self._session is not None:
                self._session.stop()
                self._session = None
            self._latest_frame = None
            return "Session stopped"

    def update_config(self, config: SessionConfig) -> str:
        with self._lock:
            if self._session is None:
                return "No active session"

            self._session.set_mode(OperatingMode(config.mode.value))
            self._session.set_waveform_kind(WaveformKind(config.waveform.value))
            self._session.radio_config.center_frequency_hz = config.center_frequency_hz
            self._session.radio_config.sample_rate_sps = config.sample_rate_sps
            self._session.radio_config.tx_gain_db = config.tx_gain_db
            self._session.radio_config.rx_gain_db = config.rx_gain_db
            self._session.radio_config.frame_size = config.frame_size
            self._session.radio_config.backend = config.backend.value
            self._session.adaptive_config.enabled = config.adaptive
            return "Config updated"

    def poll(self) -> SensingFrame | None:
        if not self.is_running or self._session is None:
            return None

        snapshot = self._session.snapshot()
        frame = serialize_frame(snapshot)
        with self._lock:
            self._latest_frame = frame
            self._frame_count += 1
        return frame

    def get_status(self) -> ServerStatus:
        uptime = time.monotonic() - self._start_time if self._start_time > 0 else 0.0
        with self._lock:
            if self._session is not None:
                snap = self._session.snapshot()
                return ServerStatus(
                    running=self.is_running,
                    mode=snap.mode_name,
                    waveform=snap.waveform_name,
                    backend=snap.backend_name,
                    uptime_s=uptime,
                    frame_count=self._frame_count,
                )
        return ServerStatus(running=False)
