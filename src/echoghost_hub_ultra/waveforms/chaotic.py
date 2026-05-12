"""Chaotic noise-like waveform generation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class ChaoticWaveformGenerator:
    """Generate a deterministic chaotic complex baseband sequence."""

    sample_rate_sps: float
    amplitude: float = 0.5
    chaotic_rate: float = 3.92
    spread_hz: float = 250_000.0
    seed: int = 13
    _rng: np.random.Generator = field(init=False, repr=False)
    _state: float = field(init=False, repr=False)
    _envelope_state: float = field(init=False, repr=False)
    _phase: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._state = float(self._rng.uniform(0.18, 0.82))
        self._envelope_state = float(self._rng.uniform(0.12, 0.88))
        self._phase = float(self._rng.uniform(0.0, math.tau))

    def _step(self, value: float) -> float:
        return self.chaotic_rate * value * (1.0 - value)

    def generate(self, num_samples: int) -> np.ndarray:
        if num_samples <= 0:
            return np.empty(0, dtype=np.complex64)

        phase = np.empty(num_samples, dtype=np.float64)
        envelope = np.empty(num_samples, dtype=np.float64)

        for index in range(num_samples):
            self._state = self._step(self._state)
            self._envelope_state = self._step(self._envelope_state)
            centered_state = 2.0 * self._state - 1.0
            centered_env = 2.0 * self._envelope_state - 1.0
            freq_offset = self.spread_hz * centered_state
            self._phase += math.tau * freq_offset / self.sample_rate_sps
            phase[index] = self._phase
            envelope[index] = 0.72 + 0.28 * math.tanh(1.8 * centered_env)

        samples = self.amplitude * envelope * np.exp(1j * phase)
        peak = float(np.max(np.abs(samples)) + 1e-12)
        samples = samples / peak * (0.98 * self.amplitude)
        return samples.astype(np.complex64, copy=False)

