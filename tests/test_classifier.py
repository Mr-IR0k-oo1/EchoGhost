from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from echoghost_hub_ultra.processing.classifier import ActivityClassifier
from echoghost_hub_ultra.processing.adaptive import WaveformAdapter
from echoghost_hub_ultra.config.presets import AdaptiveConfig, WaveformConfig


class ClassifierTests(unittest.TestCase):
    def test_classifier_returns_label_for_noisy_input(self) -> None:
        clf = ActivityClassifier()
        samples = np.random.standard_normal(1024) + 1j * np.random.standard_normal(1024)
        result = clf.classify(samples, motion_score=0.0)
        self.assertIn(result.label, {"idle", "micro-motion", "gesture", "walking", "falling"})
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_classifier_detects_high_motion(self) -> None:
        clf = ActivityClassifier()
        samples = np.exp(1j * np.linspace(0.0, 50.0 * np.pi, 1024, dtype=np.float64)).astype(np.complex64)
        result = clf.classify(samples, motion_score=0.01)
        self.assertIn(result.label, {"walking", "falling", "active"})

    def test_classifier_extracts_features(self) -> None:
        clf = ActivityClassifier()
        samples = np.ones(256, dtype=np.complex64)
        result = clf.classify(samples, motion_score=1e-6)
        self.assertIn("micro_doppler_bw", result.features)
        self.assertIn("periodicity", result.features)
        self.assertIn("zcr", result.features)
        self.assertIn("energy_var", result.features)

    def test_reset_clears_history(self) -> None:
        clf = ActivityClassifier(history_size=16)
        for _ in range(10):
            clf.classify(np.random.standard_normal(256) + 1j * np.random.standard_normal(256), 0.001)
        self.assertEqual(len(clf._feature_history), 10)
        clf.reset()
        self.assertEqual(len(clf._feature_history), 0)
        self.assertEqual(len(clf._label_history), 0)


class AdaptiveTests(unittest.TestCase):
    def test_adapter_disabled_returns_input_params(self) -> None:
        cfg = AdaptiveConfig(enabled=False)
        adapter = WaveformAdapter(cfg)
        result = adapter.update(np.ones(256, dtype=np.complex64), 0.0, WaveformConfig())
        self.assertEqual(result.spread_hz, 250_000.0)
        self.assertEqual(result.chaotic_rate, 3.92)
        self.assertEqual(result.amplitude, 0.45)

    def test_adapter_updates_params_when_enabled(self) -> None:
        cfg = AdaptiveConfig(enabled=True)
        adapter = WaveformAdapter(cfg)
        prev = None
        for i in range(10):
            samples = np.random.standard_normal(256) + 1j * np.random.standard_normal(256)
            result = adapter.update(samples, 0.001 * (i % 3 + 1), WaveformConfig())
            prev = result
        self.assertIsNotNone(prev)
        self.assertGreater(prev.snr_estimate, -20.0)

    def test_adapter_metric_history(self) -> None:
        cfg = AdaptiveConfig(enabled=True)
        adapter = WaveformAdapter(cfg)
        for i in range(8):
            samples = np.exp(1j * np.linspace(0.0, np.pi * (i + 1), 256, dtype=np.float64)).astype(np.complex64)
            result = adapter.update(samples, 0.0005, WaveformConfig())
        self.assertGreater(len(result.metric_history), 0)

    def test_reset_clears_adapter(self) -> None:
        cfg = AdaptiveConfig(enabled=True)
        adapter = WaveformAdapter(cfg)
        for _ in range(5):
            adapter.update(np.ones(256, dtype=np.complex64), 0.0, WaveformConfig())
        adapter.reset()
        self.assertEqual(len(adapter._snr_history), 0)
        self.assertEqual(len(adapter._metric_history), 0)


if __name__ == "__main__":
    unittest.main()
