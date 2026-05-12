from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ChaoticMapType(str, Enum):
    LOGISTIC = "logistic"
    HENON = "henon"
    LORENZ = "lorenz"
    KURAMOTO_SIVASHINSKY = "kuramoto_sivashinsky"


@dataclass(slots=True)
class ChaoticWaveformGenerator:
    """Multi-map chaotic noise-like complex baseband generator."""

    sample_rate_sps: float
    amplitude: float = 0.5
    chaotic_rate: float = 3.92
    spread_hz: float = 250_000.0
    seed: int = 13
    map_type: ChaoticMapType = ChaoticMapType.LOGISTIC
    _rng: np.random.Generator = field(init=False, repr=False)

    _state: float = field(init=False, repr=False)
    _envelope_state: float = field(init=False, repr=False)
    _phase: float = field(init=False, repr=False)

    _henon_x: float = field(init=False, repr=False)
    _henon_y: float = field(init=False, repr=False)

    _lorenz_x: float = field(init=False, repr=False)
    _lorenz_y: float = field(init=False, repr=False)
    _lorenz_z: float = field(init=False, repr=False)
    _lorenz_dt: float = field(init=False, repr=False)

    _ks_field: np.ndarray = field(init=False, repr=False)
    _ks_N: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._state = float(self._rng.uniform(0.18, 0.82))
        self._envelope_state = float(self._rng.uniform(0.12, 0.88))
        self._phase = float(self._rng.uniform(0.0, math.tau))
        self._henon_x = float(self._rng.uniform(-0.3, 0.3))
        self._henon_y = float(self._rng.uniform(-0.3, 0.3))
        self._lorenz_x = float(self._rng.uniform(-10.0, 10.0))
        self._lorenz_y = float(self._rng.uniform(-10.0, 10.0))
        self._lorenz_z = float(self._rng.uniform(10.0, 30.0))
        self._lorenz_dt = 0.005
        self._ks_N = 64
        self._ks_field = self._rng.uniform(-0.05, 0.05, self._ks_N).astype(np.float64)

    @staticmethod
    def _logistic_step(value: float, rate: float) -> float:
        return rate * value * (1.0 - value)

    def _logistic_generate(self, num_samples: int) -> np.ndarray:
        phase = np.empty(num_samples, dtype=np.float64)
        envelope = np.empty(num_samples, dtype=np.float64)
        for idx in range(num_samples):
            self._state = self._logistic_step(self._state, self.chaotic_rate)
            self._envelope_state = self._logistic_step(self._envelope_state, self.chaotic_rate)
            freq_offset = self.spread_hz * (2.0 * self._state - 1.0)
            self._phase += math.tau * freq_offset / self.sample_rate_sps
            phase[idx] = self._phase
            envelope[idx] = 0.72 + 0.28 * math.tanh(1.8 * (2.0 * self._envelope_state - 1.0))
        return self._normalize(self.amplitude * envelope * np.exp(1j * phase))

    def _henon_generate(self, num_samples: int) -> np.ndarray:
        a, b = 1.4, 0.3
        samples = np.empty(num_samples, dtype=np.complex128)
        for idx in range(num_samples):
            xn, yn = self._henon_x, self._henon_y
            nxt = 1.0 - a * xn * xn + yn
            self._henon_y = b * xn
            self._henon_x = nxt
            freq_offset = self.spread_hz * self._henon_x
            self._phase += math.tau * freq_offset / self.sample_rate_sps
            env = 0.6 + 0.4 * abs(self._henon_y)
            samples[idx] = env * np.exp(1j * self._phase)
        return self._normalize(self.amplitude * samples)

    def _lorenz_generate(self, num_samples: int) -> np.ndarray:
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        dt = self._lorenz_dt
        samples = np.empty(num_samples, dtype=np.complex128)
        for idx in range(num_samples):
            x, y, z = self._lorenz_x, self._lorenz_y, self._lorenz_z
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            self._lorenz_x += dt * dx
            self._lorenz_y += dt * dy
            self._lorenz_z += dt * dz
            freq_offset = 0.15 * self.spread_hz * (self._lorenz_x / 30.0)
            self._phase += math.tau * freq_offset / self.sample_rate_sps
            env = 0.5 + 0.5 * abs(self._lorenz_y) / 40.0
            samples[idx] = env * np.exp(1j * self._phase)
        return self._normalize(self.amplitude * samples)

    def _ks_generate(self, num_samples: int) -> np.ndarray:
        N = self._ks_N
        dx = 1.0
        dt = 0.05
        nu = 0.1
        samples = np.empty(num_samples, dtype=np.complex128)
        for idx in range(num_samples):
            u = self._ks_field
            uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)
            uxxxx = (
                np.roll(u, -2) - 4.0 * np.roll(u, -1) + 6.0 * u - 4.0 * np.roll(u, 1) + np.roll(u, 2)
            ) / (dx**4)
            u_term = u * (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
            self._ks_field = u + dt * (-u_term - nu * uxxxx - uxx)
            mean_field = float(np.mean(self._ks_field))
            freq_offset = self.spread_hz * mean_field * 10.0
            self._phase += math.tau * freq_offset / self.sample_rate_sps
            env = 0.5 + 0.5 * float(np.std(self._ks_field)) * 8.0
            samples[idx] = env * np.exp(1j * self._phase)
        return self._normalize(self.amplitude * samples)

    @staticmethod
    def _normalize(samples: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(samples)) + 1e-12)
        if peak > 0.0:
            samples = samples / peak
        return samples.astype(np.complex64, copy=False)

    def generate(self, num_samples: int) -> np.ndarray:
        if num_samples <= 0:
            return np.empty(0, dtype=np.complex64)

        if self.map_type is ChaoticMapType.HENON:
            return self._henon_generate(num_samples)
        if self.map_type is ChaoticMapType.LORENZ:
            return self._lorenz_generate(num_samples)
        if self.map_type is ChaoticMapType.KURAMOTO_SIVASHINSKY:
            return self._ks_generate(num_samples)
        return self._logistic_generate(num_samples)
