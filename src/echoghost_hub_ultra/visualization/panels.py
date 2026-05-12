"""Utility functions used by the GUI layer for real-time plots."""

from __future__ import annotations

import numpy as np


def normalize_heatmap(matrix: np.ndarray, floor_db: float | None = None, ceiling_db: float | None = None) -> np.ndarray:
    """Normalize a dB matrix into the 0..1 range for display."""

    if matrix.size == 0:
        return np.zeros_like(matrix, dtype=np.float32)

    matrix = np.asarray(matrix, dtype=np.float32)
    lower = float(np.min(matrix) if floor_db is None else floor_db)
    upper = float(np.max(matrix) if ceiling_db is None else ceiling_db)
    if upper <= lower:
        upper = lower + 1.0
    normalized = (matrix - lower) / (upper - lower)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def heatmap_to_rgba(matrix: np.ndarray, floor_db: float | None = None, ceiling_db: float | None = None) -> np.ndarray:
    """Convert a matrix into a flattened RGBA texture buffer."""

    normalized = normalize_heatmap(matrix, floor_db=floor_db, ceiling_db=ceiling_db)
    red = np.clip(1.8 * normalized - 0.15, 0.0, 1.0)
    green = np.clip(1.8 * (normalized - 0.28), 0.0, 1.0)
    blue = np.clip(1.35 * (1.0 - normalized), 0.0, 1.0)
    alpha = np.ones_like(red, dtype=np.float32)
    rgba = np.stack((red, green, blue, alpha), axis=-1).astype(np.float32, copy=False)
    return rgba.reshape(-1)

