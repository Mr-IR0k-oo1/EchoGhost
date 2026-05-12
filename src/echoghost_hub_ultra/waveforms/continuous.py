"""Deterministic waveform generators for active TX modes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class ContinuousToneGenerator:
    """Generate a continuous complex baseband tone."""

    frequency_hz: float
    sample_rate_sps: float
    amplitude: float = 0.5
    phase_rad: float = 0.0

    def generate(self, num_samples: int) -> np.ndarray:
        if num_samples <= 0:
            return np.empty(0, dtype=np.complex64)

        sample_index = np.arange(num_samples, dtype=np.float64)
        angular_step = math.tau * self.frequency_hz / self.sample_rate_sps
        phase = self.phase_rad + angular_step * sample_index
        samples = self.amplitude * np.exp(1j * phase)
        self.phase_rad = float((phase[-1] + angular_step) % math.tau)
        return samples.astype(np.complex64, copy=False)


@dataclass(slots=True)
class FMCWChirpGenerator:
    """Generate a repeating linear chirp in complex baseband."""

    start_frequency_hz: float
    end_frequency_hz: float
    chirp_period_s: float
    sample_rate_sps: float
    amplitude: float = 0.5
    phase_rad: float = 0.0
    sample_index: int = 0

    def generate(self, num_samples: int) -> np.ndarray:
        if num_samples <= 0:
            return np.empty(0, dtype=np.complex64)

        sample_index = np.arange(num_samples, dtype=np.float64) + float(self.sample_index)
        time_s = sample_index / self.sample_rate_sps
        sweep_t = np.mod(time_s, self.chirp_period_s)
        chirp_slope = (self.end_frequency_hz - self.start_frequency_hz) / self.chirp_period_s
        phase = self.phase_rad + math.tau * (
            self.start_frequency_hz * sweep_t + 0.5 * chirp_slope * sweep_t**2
        )
        samples = self.amplitude * np.exp(1j * phase)
        self.sample_index += num_samples
        self.phase_rad = float(np.angle(samples[-1]))
        return samples.astype(np.complex64, copy=False)


@dataclass(slots=True)
class PseudoRandomNoiseGenerator:
    """Generate band-limited pseudo-random complex noise."""

    sample_rate_sps: float
    amplitude: float = 0.5
    seed: int = 13
    smoothing_taps: int = 9
    _rng: np.random.Generator = field(init=False, repr=False)
    _kernel: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        taps = max(3, int(self.smoothing_taps))
        self._kernel = np.ones(taps, dtype=np.float64) / float(taps)

    def generate(self, num_samples: int) -> np.ndarray:
        if num_samples <= 0:
            return np.empty(0, dtype=np.complex64)

        raw = self._rng.standard_normal(num_samples) + 1j * self._rng.standard_normal(num_samples)
        shaped_real = np.convolve(raw.real, self._kernel, mode="same")
        shaped_imag = np.convolve(raw.imag, self._kernel, mode="same")
        shaped = shaped_real + 1j * shaped_imag
        peak = float(np.max(np.abs(shaped)) + 1e-12)
        shaped = shaped / peak
        return (0.95 * self.amplitude * shaped).astype(np.complex64, copy=False)

