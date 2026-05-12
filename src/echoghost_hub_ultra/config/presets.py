"""Typed configuration objects used across the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OperatingMode(str, Enum):
    """Top-level run mode for the dashboard."""

    SIMULATION = "simulation"
    ACTIVE = "active"
    PASSIVE = "passive"


class WaveformKind(str, Enum):
    """Available baseband waveform families."""

    TONE = "tone"
    CHIRP = "chirp"
    CHAOTIC = "chaotic"
    PRN = "prn"


@dataclass(slots=True)
class RadioConfig:
    center_frequency_hz: float = 915_000_000.0
    sample_rate_sps: float = 2_000_000.0
    rx_bandwidth_hz: float = 1_500_000.0
    tx_gain_db: float = 18.0
    rx_gain_db: float = 24.0
    frame_size: int = 4096
    backend: str = "simulation"
    mode: OperatingMode = OperatingMode.SIMULATION
    device_index: int = 0


@dataclass(slots=True)
class WaveformConfig:
    kind: WaveformKind = WaveformKind.TONE
    tone_frequency_hz: float = 120_000.0
    tone_amplitude: float = 0.45
    chirp_start_hz: float = -150_000.0
    chirp_end_hz: float = 150_000.0
    chirp_period_s: float = 0.010
    chaotic_spread_hz: float = 250_000.0
    chaotic_rate: float = 3.92
    prn_seed: int = 13


@dataclass(slots=True)
class ProcessingConfig:
    baseline_alpha: float = 0.05
    motion_history_size: int = 128
    breathing_history_size: int = 256
    heatmap_history_size: int = 64
    heatmap_fft_size: int = 512


@dataclass(slots=True)
class SimulationConfig:
    clutter_tone_hz: tuple[float, ...] = (-91_000.0, 18_000.0, 47_500.0)
    clutter_amplitude: tuple[float, ...] = (0.10, 0.07, 0.05)
    moving_target_frequency_hz: float = 1_250.0
    moving_target_amplitude: float = 0.22
    breathing_bpm: float = 15.0
    motion_bpm: float = 84.0
    leakage_gain: float = 0.18
    noise_std: float = 0.018
    seed: int = 2026


@dataclass(slots=True)
class DashboardConfig:
    refresh_hz: float = 20.0
    plot_history_length: int = 256
    application_title: str = "EchoGhost Hub Ultra"

