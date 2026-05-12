from __future__ import annotations

import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from echoghost_hub_ultra.processing.motion import MotionDetector
from echoghost_hub_ultra.processing.range_doppler import RangeHeatmap
from echoghost_hub_ultra.processing.vitals import BreathingEstimator


class ProcessingTests(unittest.TestCase):
    def test_motion_score_increases_for_changing_frame(self) -> None:
        detector = MotionDetector(baseline_alpha=0.05)
        static_frame = np.ones(1024, dtype=np.complex64)
        static_metrics = None
        for _ in range(5):
            static_metrics = detector.update(static_frame, 0.0)

        moving_frame = np.exp(1j * np.linspace(0.0, np.pi, 1024, dtype=np.float64)).astype(np.complex64)
        moving_metrics = detector.update(moving_frame, 1.0)

        self.assertIsNotNone(static_metrics)
        self.assertGreater(moving_metrics.motion_score, static_metrics.motion_score)

    def test_breathing_estimator_recovers_slow_phase_oscillation(self) -> None:
        estimator = BreathingEstimator(history_size=256)
        breathing_hz = 0.25
        frame_period_s = 0.25
        last_estimate = None
        for index in range(256):
            slow_time = index * frame_period_s
            phase = 1.2 * np.sin(2.0 * np.pi * breathing_hz * slow_time)
            frame = np.exp(1j * np.full(64, phase, dtype=np.float64)).astype(np.complex64)
            last_estimate = estimator.update(frame, slow_time, frame_period_s)

        self.assertIsNotNone(last_estimate)
        self.assertIsNotNone(last_estimate.breathing_bpm)
        self.assertGreater(last_estimate.breathing_bpm, 10.0)
        self.assertLess(last_estimate.breathing_bpm, 20.0)

    def test_range_heatmap_grows_over_time(self) -> None:
        heatmap = RangeHeatmap(history_size=8, fft_size=128)
        for index in range(4):
            frame = np.exp(1j * np.linspace(0.0, np.pi * (index + 1), 256, dtype=np.float64)).astype(np.complex64)
            result = heatmap.update(frame, 2_000_000.0)

        self.assertEqual(result.matrix_db.shape[1], 128)
        self.assertLessEqual(result.matrix_db.shape[0], 8)


if __name__ == "__main__":
    unittest.main()
