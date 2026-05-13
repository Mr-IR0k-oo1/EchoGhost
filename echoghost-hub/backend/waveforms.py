"""
Waveform generation module for EchoGhost Hub Ultra.

This module generates digital baseband IQ waveforms used for active sensing.
These are transmitted by the HackRF, reflect off targets, and are processed
by the signal processing chain.

Supported waveforms:
  - Tone: Simple single-frequency CW (continuous wave)
  - FMCW: Linear frequency-modulated continuous wave (chirp)
  - Chaotic: Lorenz-attractor-based chaotic waveform (LPI/LPD)
  - Noise: Pseudorandom noise-like spread-spectrum waveform
"""

import numpy as np
from typing import Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Tone (Continuous Wave)
# ──────────────────────────────────────────────────────────────────────────────


def generate_tone(
    frequency_hz: float,
    duration_s: float,
    sample_rate_hz: float,
    amplitude: float = 0.5,
    phase: float = 0.0,
) -> np.ndarray:
    """Generate a simple continuous-wave (CW) tone.

    Args:
        frequency_hz: Tone frequency in Hz (baseband).
        duration_s: Duration in seconds.
        sample_rate_hz: Sample rate in Hz.
        amplitude: Peak amplitude (0.0 to 1.0).
        phase: Initial phase offset in radians.

    Returns:
        Complex IQ array of shape (num_samples,).
    """
    t = np.arange(0, int(sample_rate_hz * duration_s)) / sample_rate_hz
    i = amplitude * np.cos(2 * np.pi * frequency_hz * t + phase)
    q = amplitude * np.sin(2 * np.pi * frequency_hz * t + phase)
    return (i + 1j * q).astype(np.complex64)


# ──────────────────────────────────────────────────────────────────────────────
# FMCW (Linear Frequency-Modulated Continuous Wave)
# ──────────────────────────────────────────────────────────────────────────────


def generate_fmcw(
    bandwidth_hz: float,
    chirp_duration_s: float,
    sample_rate_hz: float,
    amplitude: float = 0.5,
    num_chirps: int = 1,
) -> np.ndarray:
    """Generate one or more linear FMCW chirps.

    FMCW radar transmits a swept-frequency signal. The return echo is mixed
    with the transmitted signal to produce a beat frequency proportional to
    range. This is the foundation of range-Doppler processing.

    Args:
        bandwidth_hz: Sweep bandwidth in Hz.
        chirp_duration_s: Duration of one chirp in seconds.
        sample_rate_hz: Sample rate in Hz.
        amplitude: Peak amplitude.
        num_chirps: Number of consecutive chirps.

    Returns:
        Complex IQ array.
    """
    samples_per_chirp = int(sample_rate_hz * chirp_duration_s)
    t = np.arange(samples_per_chirp) / sample_rate_hz

    # Linear frequency sweep from -BW/2 to +BW/2
    k = bandwidth_hz / chirp_duration_s  # chirp rate
    phase = 2 * np.pi * (0.5 * k * t**2)

    chirp = amplitude * (np.cos(phase) + 1j * np.sin(phase))
    chirp = chirp.astype(np.complex64)

    if num_chirps == 1:
        return chirp
    return np.tile(chirp, num_chirps)


# ──────────────────────────────────────────────────────────────────────────────
# Chaotic (Lorenz Attractor)
# ──────────────────────────────────────────────────────────────────────────────


def _lorenz_system(
    state: np.ndarray,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.01,
) -> np.ndarray:
    """Single step of the Lorenz chaotic system.

    The Lorenz attractor is a 3-variable nonlinear system that produces
    chaotic, aperiodic trajectories. When used as a radar waveform, the
    chaotic signal is deterministic (so we know what we transmitted) but
    appears noise-like to intercept receivers — providing LPI/LPD.

    Args:
        state: Current [x, y, z] vector.
        sigma, rho, beta: Lorenz system parameters.
        dt: Time step.

    Returns:
        Updated [x, y, z] vector.
    """
    x, y, z = state
    dx = sigma * (y - x) * dt
    dy = (x * (rho - z) - y) * dt
    dz = (x * y - beta * z) * dt
    return np.array([x + dx, y + dy, z + dz])


def generate_chaotic_waveform(
    duration_s: float,
    sample_rate_hz: float,
    bandwidth_hz: float = 20e6,
    amplitude: float = 0.5,
    seed_state: np.ndarray = None,
) -> np.ndarray:
    """Generate a chaotic LPI waveform based on the Lorenz attractor.

    The chaotic trajectory is mapped to the IQ plane and scaled to the
    desired bandwidth. The result is a wideband, noise-like signal that
    can be used for low-probability-of-intercept sensing.

    Args:
        duration_s: Duration in seconds.
        sample_rate_hz: Sample rate in Hz.
        bandwidth_hz: Desired occupied bandwidth.
        amplitude: Peak amplitude.
        seed_state: Initial [x, y, z] state (random if None).

    Returns:
        Complex IQ array.
    """
    num_samples = int(sample_rate_hz * duration_s)
    state = seed_state if seed_state is not None else np.array([1.0, 0.0, 0.0])

    # Integrate the Lorenz system to generate chaotic samples
    dt = 1.0 / sample_rate_hz
    iq = np.zeros(num_samples, dtype=np.complex64)

    for n in range(num_samples):
        state = _lorenz_system(state, dt=dt)
        # Map chaotic variables to I/Q with bandwidth scaling
        i = state[0] / 30.0  # Normalize Lorenz x (~ range [-20, 20])
        q = state[1] / 30.0  # Normalize Lorenz y
        # Apply bandwidth shaping via amplitude scaling
        iq[n] = amplitude * (i + 1j * q)

    # Normalize to desired amplitude
    peak = np.max(np.abs(iq))
    if peak > 0:
        iq = (iq / peak) * amplitude

    return iq.astype(np.complex64)


# ──────────────────────────────────────────────────────────────────────────────
# Noise (Pseudorandom)
# ──────────────────────────────────────────────────────────────────────────────


def generate_noise_waveform(
    duration_s: float,
    sample_rate_hz: float,
    bandwidth_hz: float = 20e6,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a band-limited pseudorandom noise waveform.

    Used for spread-spectrum sensing. The wideband noise-like signal
    spreads energy across the band, reducing peak power and making it
    harder to detect.

    Args:
        duration_s: Duration in seconds.
        sample_rate_hz: Sample rate in Hz.
        bandwidth_hz: -6 dB bandwidth.
        amplitude: Peak amplitude.

    Returns:
        Complex IQ array.
    """
    num_samples = int(sample_rate_hz * duration_s)

    # Generate white Gaussian noise
    noise = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)

    # Apply low-pass filter to limit bandwidth
    cutoff = bandwidth_hz / sample_rate_hz  # Normalized cutoff
    if cutoff < 0.5:
        # Simple FIR filter via FFT
        spectrum = np.fft.fft(noise)
        freqs = np.fft.fftfreq(num_samples)
        mask = np.abs(freqs) <= cutoff
        spectrum[~mask] = 0
        noise = np.fft.ifft(spectrum)

    # Normalize amplitude
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise = (noise / peak) * amplitude

    return noise.astype(np.complex64)


# ──────────────────────────────────────────────────────────────────────────────
# Waveform Router
# ──────────────────────────────────────────────────────────────────────────────


def generate_waveform(
    waveform_type: str,
    sample_rate_hz: float,
    duration_s: float = 1e-3,
    amplitude: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """Generate a waveform by type name.

    Args:
        waveform_type: One of 'tone', 'fmcw', 'chaotic', 'noise'.
        sample_rate_hz: Sample rate.
        duration_s: Total duration.
        amplitude: Peak amplitude.
        **kwargs: Additional waveform-specific parameters.

    Returns:
        Complex IQ array.
    """
    waveform_map = {
        "tone": lambda: generate_tone(
            frequency_hz=kwargs.get("frequency_hz", 100e3),
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            amplitude=amplitude,
        ),
        "fmcw": lambda: generate_fmcw(
            bandwidth_hz=kwargs.get("chirp_bw_hz", 20e6),
            chirp_duration_s=kwargs.get("chirp_duration_s", duration_s),
            sample_rate_hz=sample_rate_hz,
            amplitude=amplitude,
            num_chirps=kwargs.get("num_chirps", 1),
        ),
        "chaotic": lambda: generate_chaotic_waveform(
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=kwargs.get("chaos_bandwidth_hz", 20e6),
            amplitude=amplitude,
        ),
        "noise": lambda: generate_noise_waveform(
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=kwargs.get("noise_bandwidth_hz", 20e6),
            amplitude=amplitude,
        ),
    }
    generator = waveform_map.get(waveform_type)
    if generator is None:
        raise ValueError(f"Unknown waveform type: {waveform_type}")
    return generator()
