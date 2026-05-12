from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OperatingMode(str, Enum):
    SIMULATION = "simulation"
    ACTIVE = "active"
    PASSIVE = "passive"


class WaveformKind(str, Enum):
    TONE = "tone"
    CHIRP = "chirp"
    CHAOTIC = "chaotic"
    CHAOTIC_HENON = "chaotic_henon"
    CHAOTIC_LORENZ = "chaotic_lorenz"
    CHAOTIC_KS = "chaotic_ks"
    PRN = "prn"


class BackendKind(str, Enum):
    SIMULATION = "simulation"
    HACKRF = "hackrf"
    SOAPY = "soapy"
    MULTI_HACKRF = "multi_hackrf"


class SessionConfig(BaseModel):
    mode: OperatingMode = OperatingMode.SIMULATION
    waveform: WaveformKind = WaveformKind.TONE
    backend: BackendKind = BackendKind.SIMULATION
    center_frequency_hz: float = 915_000_000.0
    sample_rate_sps: float = 2_000_000.0
    frame_size: int = 4096
    tx_gain_db: float = 18.0
    rx_gain_db: float = 24.0
    adaptive: bool = False


class MotionData(BaseModel):
    score: float = 0.0
    label: str = "idle"
    confidence: float = 0.0


class BreathingData(BaseModel):
    bpm: float | None = None
    confidence: float = 0.0


class PositionData(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    intensity: float = 0.0
    label: str = ""


class SensingFrame(BaseModel):
    t: float = 0.0
    mode: str = "simulation"
    waveform: str = "tone"
    backend: str = "simulation"
    status: str = ""
    motion: MotionData = Field(default_factory=MotionData)
    breathing: BreathingData = Field(default_factory=BreathingData)
    ambient_energy_db: float = -120.0
    spectrum: list[float] = Field(default_factory=list)
    heatmap_rows: int = 0
    heatmap_cols: int = 0
    heatmap_data: list[float] = Field(default_factory=list)
    positions: list[dict[str, float]] = Field(default_factory=list)


class ServerStatus(BaseModel):
    running: bool = False
    mode: str = "simulation"
    waveform: str = "tone"
    backend: str = "simulation"
    uptime_s: float = 0.0
    frame_count: int = 0
