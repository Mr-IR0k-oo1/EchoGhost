"""Very small vital-sign estimation helpers built on phase history."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BreathingEstimate:
    """Estimated breathing rate from phase slow-time history."""

    timestamp_s: float
    breathing_hz: float | None
    breathing_bpm: float | None
    confidence: float


class BreathingEstimator:
    """Estimate breathing rate from the slow-time phase of a complex frame."""

    def __init__(self, history_size: int = 256, min_hz: float = 0.08, max_hz: float = 0.7) -> None:
        self.min_hz = float(min_hz)
        self.max_hz = float(max_hz)
        self.phase_history: deque[float] = deque(maxlen=int(history_size))
        self.bpm_history: deque[float] = deque(maxlen=int(history_size))

    def update(self, samples: np.ndarray, timestamp_s: float, frame_period_s: float) -> BreathingEstimate:
        frame = np.asarray(samples, dtype=np.complex64)
        if frame.size == 0:
            return BreathingEstimate(
                timestamp_s=timestamp_s,
                breathing_hz=None,
                breathing_bpm=None,
                confidence=0.0,
            )

        phase_sample = float(np.angle(np.mean(frame)))
        self.phase_history.append(phase_sample)

        if len(self.phase_history) < 24 or frame_period_s <= 0.0:
            return BreathingEstimate(
                timestamp_s=timestamp_s,
                breathing_hz=None,
                breathing_bpm=None,
                confidence=0.0,
            )

        series = np.unwrap(np.asarray(self.phase_history, dtype=np.float64))
        trend = np.linspace(series[0], series[-1], series.size)
        detrended = series - trend
        window = np.hanning(series.size)
        spectrum = np.abs(np.fft.rfft(detrended * window))
        frequencies = np.fft.rfftfreq(series.size, d=frame_period_s)
        band = (frequencies >= self.min_hz) & (frequencies <= self.max_hz)
        if not np.any(band):
            return BreathingEstimate(
                timestamp_s=timestamp_s,
                breathing_hz=None,
                breathing_bpm=None,
                confidence=0.0,
            )

        band_spectrum = spectrum[band]
        band_frequencies = frequencies[band]
        peak_index = int(np.argmax(band_spectrum))
        breathing_hz = float(band_frequencies[peak_index])
        breathing_bpm = 60.0 * breathing_hz
        confidence = float(band_spectrum[peak_index] / (np.mean(band_spectrum) + 1e-9))
        self.bpm_history.append(breathing_bpm)
        return BreathingEstimate(
            timestamp_s=timestamp_s,
            breathing_hz=breathing_hz,
            breathing_bpm=breathing_bpm,
            confidence=confidence,
        )

