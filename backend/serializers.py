from __future__ import annotations

import math
import os
import sys

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from echoghost_hub_ultra.radio.session import DashboardSnapshot

from .config import MotionData, BreathingData, PositionData, SensingFrame


def _downsample_spectrum(spectrum_db: np.ndarray, target: int = 256) -> list[float]:
    arr = np.asarray(spectrum_db, dtype=np.float32)
    if arr.size == 0:
        return []
    if arr.size <= target:
        return arr.tolist()
    indices = np.linspace(0, arr.size - 1, target, dtype=int)
    return arr[indices].tolist()


def _downsample_heatmap(matrix_db: np.ndarray, max_rows: int = 32, max_cols: int = 64) -> tuple[int, int, list[float]]:
    arr = np.asarray(matrix_db, dtype=np.float32)
    if arr.size == 0 or arr.ndim < 2:
        return (0, 0, [])
    r, c = arr.shape
    row_step = max(1, r // max_rows)
    col_step = max(1, c // max_cols)
    downsampled = arr[::row_step, ::col_step]
    return downsampled.shape[0], downsampled.shape[1], downsampled.ravel().tolist()


def _estimate_positions(spectrum_db: np.ndarray, motion_score: float, motion_label: str) -> list[dict[str, float]]:
    """Derive approximate 3D positions from spectrum energy peaks."""
    arr = np.asarray(spectrum_db, dtype=np.float32)
    if arr.size < 4 or motion_score < 1e-5:
        return []

    energy = np.abs(arr)
    threshold = float(np.mean(energy)) + 1.5 * float(np.std(energy))
    peaks = []
    for i in range(1, arr.size - 1):
        if energy[i] > threshold and energy[i] > energy[i - 1] and energy[i] > energy[i + 1]:
            normalised_i = i / max(arr.size - 1, 1)
            x = 2.0 * normalised_i - 1.0
            intensity = float(np.clip((energy[i] - threshold) / (energy.max() - threshold + 1e-12), 0.0, 1.0))
            z = float(np.clip(motion_score * 10.0, 0.0, 2.0))
            peaks.append({"x": x * 3.0, "y": float(np.sin(i * 0.5) * 0.5), "z": z, "intensity": intensity, "label": motion_label})
            if len(peaks) >= 5:
                break
    return peaks


def serialize_frame(snapshot: DashboardSnapshot) -> SensingFrame:
    spectrum = _downsample_spectrum(snapshot.spectrum_db, target=256)
    hr, hc, hdata = _downsample_heatmap(snapshot.heatmap_result.matrix_db, max_rows=32, max_cols=64)
    positions = _estimate_positions(snapshot.spectrum_db, snapshot.motion_score, snapshot.motion_label)

    return SensingFrame(
        t=snapshot.timestamp_s,
        mode=snapshot.mode_name,
        waveform=snapshot.waveform_name,
        backend=snapshot.backend_name,
        status=snapshot.status_text,
        motion=MotionData(
            score=snapshot.motion_score,
            label=snapshot.motion_label,
            confidence=snapshot.motion_confidence,
        ),
        breathing=BreathingData(
            bpm=snapshot.breathing_bpm,
            confidence=snapshot.breathing_confidence,
        ),
        ambient_energy_db=snapshot.ambient_energy_db,
        spectrum=spectrum,
        heatmap_rows=hr,
        heatmap_cols=hc,
        heatmap_data=hdata,
        positions=positions,
    )
