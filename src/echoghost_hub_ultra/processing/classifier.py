from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


ACTIVITY_LABELS = (
    "idle",
    "micro-motion",
    "gesture",
    "walking",
    "falling",
)


@dataclass(slots=True)
class ActivityResult:
    label: str
    confidence: float
    features: dict[str, float]


class ActivityClassifier:
    """Multi-class activity classifier using hand-crafted RF features.

    Features extracted per frame:
      - micro-doppler bandwidth (spectral spread)
      - periodicity strength (autocorrelation peak)
      - zero-crossing rate of phase
      - short-term energy variance
      - motion score (from MotionDetector)

    Uses threshold-based classification by default. If scikit-learn is
    available, a RandomForest can be trained via ``fit()``.
    """

    def __init__(self, history_size: int = 64) -> None:
        self.history_size = int(history_size)
        self._feature_history: deque[dict[str, float]] = deque(maxlen=self.history_size)
        self._label_history: deque[str] = deque(maxlen=self.history_size)
        self._classifier = None
        self._is_fitted = False

    def _extract_features(self, samples: np.ndarray, motion_score: float) -> dict[str, float]:
        frame = np.asarray(samples, dtype=np.complex64)
        if frame.size == 0:
            return {"micro_doppler_bw": 0.0, "periodicity": 0.0, "zcr": 0.0, "energy_var": 0.0, "motion_score": motion_score}

        spectrum = np.fft.fft(frame)
        power = np.abs(spectrum) ** 2
        total_power = float(np.sum(power) + 1e-12)
        freqs = np.fft.fftfreq(frame.size)
        centroid = float(np.sum(freqs * power) / total_power)
        spread = float(np.sqrt(np.sum((freqs - centroid) ** 2 * power) / total_power))
        micro_doppler_bw = spread * 1000.0

        autocorr = np.correlate(np.abs(frame), np.abs(frame), mode="full")
        autocorr = autocorr[autocorr.size // 2 :]
        if autocorr.size > 2:
            side_peaks = autocorr[1:]
            periodicity = float(np.max(side_peaks) / (autocorr[0] + 1e-12))
        else:
            periodicity = 0.0

        phase = np.angle(frame)
        phase_diff = np.diff(np.unwrap(phase))
        if phase_diff.size > 1:
            zcr = float(np.sum(np.abs(np.diff(np.sign(phase_diff))) > 0)) / float(phase_diff.size)
        else:
            zcr = 0.0

        energy_var = float(np.var(np.abs(frame)))

        return {
            "micro_doppler_bw": float(micro_doppler_bw),
            "periodicity": float(periodicity),
            "zcr": float(zcr),
            "energy_var": float(energy_var),
            "motion_score": float(motion_score),
        }

    def _threshold_classify(self, features: dict[str, float]) -> tuple[str, float]:
        ms = features["motion_score"]
        bw = features["micro_doppler_bw"]
        zcr = features["zcr"]
        periodicity = features["periodicity"]

        if ms < 1e-5:
            return "idle", 0.8
        if ms < 1e-4:
            return "micro-motion", 0.65

        if ms < 4e-4 and zcr > 0.15:
            return "gesture", 0.55 + 0.3 * min(zcr, 0.5)

        if bw > 300.0 and zcr > 0.3:
            return "falling", 0.5 + 0.4 * min(bw / 1000.0, 1.0)

        if ms >= 4e-4 and periodicity < 0.4:
            return "walking", min(0.95, 0.5 + ms * 200.0)

        return "active", 0.5

    def _ml_classify(self, features: dict[str, float]) -> tuple[str, float]:
        if self._classifier is None or not self._is_fitted:
            return self._threshold_classify(features)
        X = np.array([[features[k] for k in ("micro_doppler_bw", "periodicity", "zcr", "energy_var", "motion_score")]])
        probs = self._classifier.predict_proba(X)[0]
        best_idx = int(np.argmax(probs))
        label = self._classifier.classes_[best_idx]
        confidence = float(probs[best_idx])
        return str(label), confidence

    def classify(self, samples: np.ndarray, motion_score: float) -> ActivityResult:
        features = self._extract_features(samples, motion_score)
        self._feature_history.append(features)

        if self._is_fitted:
            label, confidence = self._ml_classify(features)
        else:
            label, confidence = self._threshold_classify(features)

        self._label_history.append(label)
        return ActivityResult(label=label, confidence=confidence, features=features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train a RandomForest classifier on extracted feature vectors."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            self._classifier = RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=13, class_weight="balanced"
            )
            self._classifier.fit(X, y)
            self._is_fitted = True
        except ImportError:
            self._is_fitted = False

    def reset(self) -> None:
        self._feature_history.clear()
        self._label_history.clear()
