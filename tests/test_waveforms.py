from __future__ import annotations

import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from echoghost_hub_ultra.config.presets import WaveformConfig, WaveformKind
from echoghost_hub_ultra.waveforms.factory import create_waveform_generator


class WaveformTests(unittest.TestCase):
    def test_tone_generator_has_stable_amplitude(self) -> None:
        generator = create_waveform_generator(
            WaveformConfig(kind=WaveformKind.TONE, tone_frequency_hz=1000.0, tone_amplitude=0.5),
            100_000.0,
        )
        samples = generator.generate(512)
        self.assertEqual(samples.dtype, np.complex64)
        self.assertLess(abs(float(np.max(np.abs(samples))) - 0.5), 0.05)

    def test_chaotic_generator_is_deterministic(self) -> None:
        config = WaveformConfig(kind=WaveformKind.CHAOTIC, tone_amplitude=0.4, prn_seed=42)
        generator_a = create_waveform_generator(config, 2_000_000.0)
        generator_b = create_waveform_generator(config, 2_000_000.0)
        samples_a = generator_a.generate(256)
        samples_b = generator_b.generate(256)
        self.assertTrue(np.allclose(samples_a, samples_b))

    def test_prn_generator_is_bounded(self) -> None:
        generator = create_waveform_generator(
            WaveformConfig(kind=WaveformKind.PRN, tone_amplitude=0.3, prn_seed=7),
            1_000_000.0,
        )
        samples = generator.generate(256)
        self.assertLessEqual(float(np.max(np.abs(samples))), 0.35)


if __name__ == "__main__":
    unittest.main()
