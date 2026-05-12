"""Session manager that ties together the radio backend and processors."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..config.presets import DashboardConfig, OperatingMode, ProcessingConfig, RadioConfig, SimulationConfig, WaveformConfig, WaveformKind
from ..processing.motion import MotionDetector
from ..processing.range_doppler import RangeHeatmap, RangeHeatmapResult
from ..processing.vitals import BreathingEstimator
from ..waveforms.factory import create_waveform_generator
from .backend import BackendUnavailableError, IQFrame, RadioBackend
from .hackrf_backend import HackRFBackend
from .simulator import SimulationBackend


@dataclass(slots=True)
class DashboardSnapshot:
    """Latest processed frame for the dashboard."""

    timestamp_s: float
    backend_name: str
    mode_name: str
    waveform_name: str
    center_frequency_hz: float
    sample_rate_sps: float
    status_text: str
    motion_score: float
    motion_label: str
    motion_confidence: float
    breathing_bpm: float | None
    breathing_confidence: float
    spectrum_frequency_hz: np.ndarray
    spectrum_db: np.ndarray
    motion_history: np.ndarray
    breathing_history_bpm: np.ndarray
    heatmap_result: RangeHeatmapResult
    iq_preview: np.ndarray = field(repr=False)


class RFSession:
    """Background session that drives TX/RX and feature extraction."""

    def __init__(
        self,
        radio_config: RadioConfig | None = None,
        waveform_config: WaveformConfig | None = None,
        processing_config: ProcessingConfig | None = None,
        dashboard_config: DashboardConfig | None = None,
        simulation_config: SimulationConfig | None = None,
    ) -> None:
        self.radio_config = radio_config or RadioConfig()
        self.waveform_config = waveform_config or WaveformConfig()
        self.processing_config = processing_config or ProcessingConfig()
        self.dashboard_config = dashboard_config or DashboardConfig()
        self.simulation_config = simulation_config or SimulationConfig()
        self._mode = self.radio_config.mode
        self._waveform_kind = WaveformKind(self.waveform_config.kind)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._backend: RadioBackend | None = None
        self._latest_snapshot: DashboardSnapshot | None = None
        self._status_text = "Initialising"
        self._frame_period_s = 1.0 / max(self.dashboard_config.refresh_hz, 1e-6)
        self._motion_detector = MotionDetector(
            baseline_alpha=self.processing_config.baseline_alpha,
            history_size=self.processing_config.motion_history_size,
        )
        self._breathing_estimator = BreathingEstimator(history_size=self.processing_config.breathing_history_size)
        self._heatmap = RangeHeatmap(
            history_size=self.processing_config.heatmap_history_size,
            fft_size=self.processing_config.heatmap_fft_size,
        )
        self._waveform_generator = create_waveform_generator(self.waveform_config, self.radio_config.sample_rate_sps)
        self._start_time = time.monotonic()

    @property
    def mode(self) -> OperatingMode:
        return self._mode

    @property
    def waveform_kind(self) -> WaveformKind:
        return self._waveform_kind

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_mode(self, mode: OperatingMode | str) -> None:
        self._mode = mode if isinstance(mode, OperatingMode) else OperatingMode(mode)
        self._status_text = f"Mode set to {self._mode.value}"

    def set_waveform_kind(self, waveform_kind: WaveformKind | str) -> None:
        self._waveform_kind = waveform_kind if isinstance(waveform_kind, WaveformKind) else WaveformKind(waveform_kind)
        self.waveform_config.kind = self._waveform_kind
        self._waveform_generator = create_waveform_generator(self.waveform_config, self.radio_config.sample_rate_sps)
        self._status_text = f"Waveform set to {self._waveform_kind.value}"

    def start(self) -> None:
        if self.is_running:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="echoghost-session", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._backend is not None:
            self._backend.close()
        self._backend = None

    def snapshot(self) -> DashboardSnapshot:
        with self._lock:
            if self._latest_snapshot is not None:
                return self._latest_snapshot
        return self._empty_snapshot("waiting for first frame")

    def _run(self) -> None:
        backend = self._build_backend()
        self._backend = backend
        try:
            backend.open()
        except BackendUnavailableError as exc:
            self._status_text = f"Hardware unavailable, falling back to simulation: {exc}"
            backend = SimulationBackend(self.radio_config, self.simulation_config)
            backend.open()
            self._backend = backend

        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            frame_size = self.radio_config.frame_size

            if self._mode is OperatingMode.ACTIVE:
                tx_samples = self._waveform_generator.generate(frame_size)
                try:
                    backend.transmit(tx_samples)
                except BackendUnavailableError as exc:
                    self._status_text = f"TX failed, switching to simulation: {exc}"
                    backend.close()
                    backend = SimulationBackend(self.radio_config, self.simulation_config)
                    backend.open()
                    self._backend = backend

            frame = backend.receive(frame_size, time_step_s=self._frame_period_s)
            snapshot = self._process_frame(frame)
            with self._lock:
                self._latest_snapshot = snapshot

            next_tick += self._frame_period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

            _ = loop_start

        backend.close()

    def _build_backend(self) -> RadioBackend:
        backend_name = self.radio_config.backend.lower().strip()
        if backend_name in {"hackrf", "hardware", "soapy"}:
            try:
                return HackRFBackend(self.radio_config)
            except BackendUnavailableError:
                return SimulationBackend(self.radio_config, self.simulation_config)
        return SimulationBackend(self.radio_config, self.simulation_config)

    def _process_frame(self, frame: IQFrame) -> DashboardSnapshot:
        timestamp_s = time.monotonic() - self._start_time
        motion_metrics = self._motion_detector.update(frame.samples, timestamp_s)
        breathing = self._breathing_estimator.update(frame.samples, timestamp_s, self._frame_period_s)
        heatmap_result = self._heatmap.update(frame.samples, frame.sample_rate_sps)
        spectrum_frequency_hz = heatmap_result.frequency_axis_hz
        spectrum_db = heatmap_result.matrix_db[-1] if heatmap_result.matrix_db.size else np.zeros(self.processing_config.heatmap_fft_size, dtype=np.float32)
        motion_history = np.asarray(self._motion_detector.motion_history, dtype=np.float32)
        breathing_history_bpm = np.asarray(self._breathing_estimator.bpm_history, dtype=np.float32)
        iq_preview = np.asarray(frame.samples[: min(256, frame.samples.size)], dtype=np.complex64)

        return DashboardSnapshot(
            timestamp_s=timestamp_s,
            backend_name=frame.backend_name,
            mode_name=self._mode.value,
            waveform_name=self._waveform_kind.value,
            center_frequency_hz=frame.center_frequency_hz,
            sample_rate_sps=frame.sample_rate_sps,
            status_text=self._status_text,
            motion_score=motion_metrics.motion_score,
            motion_label=motion_metrics.motion_label,
            motion_confidence=motion_metrics.motion_confidence,
            breathing_bpm=breathing.breathing_bpm,
            breathing_confidence=breathing.confidence,
            spectrum_frequency_hz=spectrum_frequency_hz,
            spectrum_db=spectrum_db,
            motion_history=motion_history,
            breathing_history_bpm=breathing_history_bpm,
            heatmap_result=heatmap_result,
            iq_preview=iq_preview,
        )

    def _empty_snapshot(self, status_text: str) -> DashboardSnapshot:
        fft_size = self.processing_config.heatmap_fft_size
        return DashboardSnapshot(
            timestamp_s=0.0,
            backend_name="simulation",
            mode_name=self._mode.value,
            waveform_name=self._waveform_kind.value,
            center_frequency_hz=self.radio_config.center_frequency_hz,
            sample_rate_sps=self.radio_config.sample_rate_sps,
            status_text=status_text,
            motion_score=0.0,
            motion_label="idle",
            motion_confidence=0.0,
            breathing_bpm=None,
            breathing_confidence=0.0,
            spectrum_frequency_hz=np.zeros(fft_size, dtype=np.float32),
            spectrum_db=np.zeros(fft_size, dtype=np.float32),
            motion_history=np.zeros(1, dtype=np.float32),
            breathing_history_bpm=np.zeros(1, dtype=np.float32),
            heatmap_result=RangeHeatmapResult(
                matrix_db=np.zeros((0, fft_size), dtype=np.float32),
                frequency_axis_hz=np.zeros(fft_size, dtype=np.float32),
                min_db=-120.0,
                max_db=0.0,
            ),
            iq_preview=np.zeros(1, dtype=np.complex64),
        )

