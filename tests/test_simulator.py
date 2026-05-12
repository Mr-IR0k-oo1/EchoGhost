from __future__ import annotations

import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from echoghost_hub_ultra.config.presets import RadioConfig, SimulationConfig
from echoghost_hub_ultra.radio.simulator import SimulationBackend


class SimulationBackendTests(unittest.TestCase):
    def test_receive_and_transmit_produce_frames(self) -> None:
        backend = SimulationBackend(
            RadioConfig(sample_rate_sps=1_000_000.0, frame_size=256),
            SimulationConfig(seed=11),
        )
        backend.open()
        try:
            first_frame = backend.receive(256, time_step_s=0.05)
            tx = np.exp(1j * np.linspace(0.0, 2.0 * np.pi, 256, dtype=np.float64)).astype(np.complex64)
            transmitted = backend.transmit(tx)
            second_frame = backend.receive(256, time_step_s=0.05)
        finally:
            backend.close()

        self.assertEqual(first_frame.samples.size, 256)
        self.assertEqual(second_frame.samples.size, 256)
        self.assertEqual(transmitted, 256)
        self.assertIn("breathing_bpm", first_frame.metadata)
        self.assertFalse(np.allclose(first_frame.samples, second_frame.samples))


if __name__ == "__main__":
    unittest.main()
