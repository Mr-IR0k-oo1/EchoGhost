"""Synthetic RF scene used for development and automated tests."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..config.presets import RadioConfig, SimulationConfig
from .backend import IQFrame, RadioBackend


@dataclass(slots=True)
class SimulationSceneState:
    """Internal state for the synthetic scene."""

    elapsed_s: float = 0.0
    moving_phase_rad: float = 0.0


class SimulationBackend(RadioBackend):
    """Generate synthetic motion, leakage, and clutter signatures."""

    name = "simulation"

    def __init__(
        self,
        radio_config: RadioConfig,
        simulation_config: SimulationConfig | None = None,
    ) -> None:
        self.radio_config = radio_config
        self.simulation_config = simulation_config or SimulationConfig()
        self._rng = np.random.default_rng(self.simulation_config.seed)
        self._state = SimulationSceneState()
        self._last_tx: np.ndarray | None = None
        self._opened = False
        self._static_phases = self._rng.uniform(0.0, math.tau, size=len(self.simulation_config.clutter_tone_hz))

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False
        self._last_tx = None

    def transmit(self, samples: np.ndarray) -> int:
        self._last_tx = np.asarray(samples, dtype=np.complex64).copy()
        return int(self._last_tx.size)

    def receive(self, num_samples: int, time_step_s: float | None = None) -> IQFrame:
        if not self._opened:
            self.open()

        sample_rate = float(self.radio_config.sample_rate_sps)
        dt = float(time_step_s) if time_step_s is not None else num_samples / sample_rate
        sample_index = np.arange(num_samples, dtype=np.float64)
        t = sample_index / sample_rate
        elapsed = self._state.elapsed_s

        clutter = np.zeros(num_samples, dtype=np.complex128)
        for tone_hz, amplitude, phase in zip(
            self.simulation_config.clutter_tone_hz,
            self.simulation_config.clutter_amplitude,
            self._static_phases,
        ):
            clutter += amplitude * np.exp(1j * (math.tau * tone_hz * t + phase + 0.05 * elapsed))

        breathing_hz = self.simulation_config.breathing_bpm / 60.0
        motion_hz = self.simulation_config.motion_bpm / 60.0
        breathing_mod = 0.18 * np.sin(math.tau * breathing_hz * elapsed)
        motion_mod = 0.22 * np.sin(math.tau * motion_hz * elapsed)
        moving_target = self.simulation_config.moving_target_amplitude * (1.0 + breathing_mod + motion_mod)
        moving_target *= np.exp(1j * (math.tau * self.simulation_config.moving_target_frequency_hz * t + self._state.moving_phase_rad))

        leakage = np.zeros(num_samples, dtype=np.complex128)
        if self._last_tx is not None and self._last_tx.size > 0:
            tx = np.asarray(self._last_tx, dtype=np.complex128)
            if tx.size < num_samples:
                repeats = int(np.ceil(num_samples / tx.size))
                tx = np.tile(tx, repeats)
            tx = tx[:num_samples]
            leakage = self.simulation_config.leakage_gain * (tx + 0.35 * np.roll(tx, 5))

        noise = self.simulation_config.noise_std * (
            self._rng.standard_normal(num_samples) + 1j * self._rng.standard_normal(num_samples)
        )

        samples = clutter + moving_target + leakage + noise
        self._state.elapsed_s += dt
        self._state.moving_phase_rad = float((self._state.moving_phase_rad + math.tau * 1.2 * dt) % math.tau)

        return IQFrame(
            samples=samples.astype(np.complex64, copy=False),
            timestamp_s=time.monotonic(),
            backend_name=self.name,
            center_frequency_hz=self.radio_config.center_frequency_hz,
            sample_rate_sps=self.radio_config.sample_rate_sps,
            metadata={
                "breathing_bpm": float(self.simulation_config.breathing_bpm),
                "motion_bpm": float(self.simulation_config.motion_bpm),
                "elapsed_s": float(self._state.elapsed_s),
            },
        )

