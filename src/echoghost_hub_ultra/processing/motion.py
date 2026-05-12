"""Baseline subtraction and motion scoring."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class MotionMetrics:
    """Summary of the motion features extracted from a frame."""

    timestamp_s: float
    residual_energy: float
    magnitude_variance: float
    phase_variance: float
    motion_score: float
    motion_label: str
    motion_confidence: float
    baseline_energy: float


class MotionDetector:
    """Track a running complex baseline and detect deviations from it."""

    def __init__(self, baseline_alpha: float = 0.05, history_size: int = 128) -> None:
        self.baseline_alpha = float(baseline_alpha)
        self._baseline_frame: np.ndarray | None = None
        self.motion_history: deque[float] = deque(maxlen=int(history_size))

    def update(self, samples: np.ndarray, timestamp_s: float) -> MotionMetrics:
        frame = np.asarray(samples, dtype=np.complex64)
        if frame.size == 0:
            return MotionMetrics(
                timestamp_s=timestamp_s,
                residual_energy=0.0,
                magnitude_variance=0.0,
                phase_variance=0.0,
                motion_score=0.0,
                motion_label="idle",
                motion_confidence=0.0,
                baseline_energy=0.0,
            )

        if self._baseline_frame is None or self._baseline_frame.shape != frame.shape:
            self._baseline_frame = frame.copy()
        else:
            alpha = self.baseline_alpha
            self._baseline_frame = (1.0 - alpha) * self._baseline_frame + alpha * frame

        residual = frame - self._baseline_frame
        residual_energy = float(np.mean(np.abs(residual) ** 2))
        magnitude_variance = float(np.var(np.abs(frame)))
        phase = np.unwrap(np.angle(frame))
        phase_delta = np.diff(phase)
        phase_variance = float(np.var(phase_delta)) if phase_delta.size else 0.0
        baseline_energy = float(np.mean(np.abs(self._baseline_frame) ** 2))
        motion_score = residual_energy + 0.45 * magnitude_variance + 0.25 * phase_variance
        self.motion_history.append(motion_score)

        motion_label = self._classify_motion(motion_score)
        motion_confidence = float(1.0 - np.exp(-motion_score / (baseline_energy + 1e-6)))
        return MotionMetrics(
            timestamp_s=timestamp_s,
            residual_energy=residual_energy,
            magnitude_variance=magnitude_variance,
            phase_variance=phase_variance,
            motion_score=motion_score,
            motion_label=motion_label,
            motion_confidence=motion_confidence,
            baseline_energy=baseline_energy,
        )

    def _classify_motion(self, motion_score: float) -> str:
        if motion_score < 1e-5:
            return "quiet"
        if motion_score < 1e-4:
            return "micro-motion"
        if motion_score < 4e-4:
            return "gesture"
        if motion_score < 1.2e-3:
            return "walking"
        return "active"

