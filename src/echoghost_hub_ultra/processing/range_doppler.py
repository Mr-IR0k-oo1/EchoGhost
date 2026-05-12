"""Rolling spectral heat map used as a range/Doppler proxy in v1."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RangeHeatmapResult:
    """History matrix plus axes for visualization."""

    matrix_db: np.ndarray
    frequency_axis_hz: np.ndarray
    min_db: float
    max_db: float


class RangeHeatmap:
    """Build a rolling FFT history that can be rendered as a heat map."""

    def __init__(self, history_size: int = 64, fft_size: int = 512) -> None:
        self.history_size = int(history_size)
        self.fft_size = int(fft_size)
        self._history: deque[np.ndarray] = deque(maxlen=self.history_size)

    def update(self, samples: np.ndarray, sample_rate_sps: float) -> RangeHeatmapResult:
        frame = np.asarray(samples, dtype=np.complex64)
        if frame.size == 0:
            empty = np.zeros((0, self.fft_size), dtype=np.float32)
            freq_axis = np.fft.fftshift(np.fft.fftfreq(self.fft_size, d=1.0 / sample_rate_sps))
            return RangeHeatmapResult(matrix_db=empty, frequency_axis_hz=freq_axis, min_db=-120.0, max_db=0.0)

        spectrum = np.fft.fftshift(np.fft.fft(frame, n=self.fft_size))
        spectrum_db = 20.0 * np.log10(np.abs(spectrum) + 1e-12)
        self._history.append(spectrum_db.astype(np.float32, copy=False))
        matrix = np.vstack(self._history) if self._history else spectrum_db[np.newaxis, :]
        freq_axis = np.fft.fftshift(np.fft.fftfreq(self.fft_size, d=1.0 / sample_rate_sps))
        return RangeHeatmapResult(
            matrix_db=matrix.astype(np.float32, copy=False),
            frequency_axis_hz=freq_axis.astype(np.float32, copy=False),
            min_db=float(np.min(matrix)),
            max_db=float(np.max(matrix)),
        )

