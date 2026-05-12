"""Waveform factory for dashboard-controlled TX modes."""

from __future__ import annotations

from .chaotic import ChaoticWaveformGenerator
from .continuous import ContinuousToneGenerator, FMCWChirpGenerator, PseudoRandomNoiseGenerator
from ..config.presets import WaveformConfig, WaveformKind


def _coerce_kind(kind: WaveformKind | str) -> WaveformKind:
    return kind if isinstance(kind, WaveformKind) else WaveformKind(kind)


def create_waveform_generator(config: WaveformConfig, sample_rate_sps: float):
    """Create a waveform generator from the active waveform config."""

    kind = _coerce_kind(config.kind)
    if kind is WaveformKind.TONE:
        return ContinuousToneGenerator(
            frequency_hz=config.tone_frequency_hz,
            sample_rate_sps=sample_rate_sps,
            amplitude=config.tone_amplitude,
        )
    if kind is WaveformKind.CHIRP:
        return FMCWChirpGenerator(
            start_frequency_hz=config.chirp_start_hz,
            end_frequency_hz=config.chirp_end_hz,
            chirp_period_s=config.chirp_period_s,
            sample_rate_sps=sample_rate_sps,
            amplitude=config.tone_amplitude,
        )
    if kind is WaveformKind.CHAOTIC:
        return ChaoticWaveformGenerator(
            sample_rate_sps=sample_rate_sps,
            amplitude=config.tone_amplitude,
            chaotic_rate=config.chaotic_rate,
            spread_hz=config.chaotic_spread_hz,
            seed=config.prn_seed,
        )
    if kind is WaveformKind.PRN:
        return PseudoRandomNoiseGenerator(
            sample_rate_sps=sample_rate_sps,
            amplitude=config.tone_amplitude,
            seed=config.prn_seed,
        )
    raise ValueError(f"Unsupported waveform kind: {kind}")

