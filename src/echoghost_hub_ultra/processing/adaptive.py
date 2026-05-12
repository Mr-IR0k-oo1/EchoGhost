from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config.presets import AdaptiveConfig, WaveformConfig


@dataclass(slots=True)
class AdaptationResult:
    spread_hz: float
    chaotic_rate: float
    amplitude: float
    snr_estimate: float
    metric_history: tuple[float, ...]


class WaveformAdapter:
    """Real-time waveform parameter optimizer using environment feedback.

    Uses a simple hill-climbing strategy: perturb a parameter, measure
    the SNR / motion sensitivity, and move in the direction of improvement.

    Tunes three parameters:
      - spread_hz:    chaotic bandwidth
      - chaotic_rate: logistic map rate parameter (3.5-3.99)
      - amplitude:    TX amplitude
    """

    def __init__(self, adaptive_config: AdaptiveConfig | None = None) -> None:
        self.config = adaptive_config or AdaptiveConfig()
        self._snr_history: deque[float] = deque(maxlen=self.config.snr_window_size)
        self._metric_history: list[float] = []
        self._spread = self.config.spread_hz_min
        self._rate = self.config.chaotic_rate_min
        self._amplitude = self.config.amplitude_min
        self._step_index = 0
        self._best_metric = -1e9
        self._best_params = (self._spread, self._rate, self._amplitude)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.config.enabled = bool(value)

    @staticmethod
    def _estimate_snr(frame: np.ndarray) -> float:
        power = np.abs(frame) ** 2
        signal_power = float(np.mean(power))
        noise_power = float(np.var(power))
        if noise_power <= 0.0:
            return 40.0
        snr = 10.0 * np.log10(signal_power / (noise_power + 1e-12))
        return float(np.clip(snr, -20.0, 60.0))

    def _compute_metric(self, snr: float, motion_score: float, dynamic_range: float) -> float:
        return 0.5 * snr + 0.3 * motion_score * 1000.0 + 0.2 * dynamic_range

    def update(
        self, rx_frame: np.ndarray, motion_score: float, current_config: WaveformConfig
    ) -> AdaptationResult:
        if not self.config.enabled:
            return AdaptationResult(
                spread_hz=current_config.chaotic_spread_hz,
                chaotic_rate=current_config.chaotic_rate,
                amplitude=current_config.tone_amplitude,
                snr_estimate=0.0,
                metric_history=(),
            )

        snr = self._estimate_snr(rx_frame)
        self._snr_history.append(snr)

        dynamic_range = float(20.0 * np.log10(np.max(np.abs(rx_frame)) + 1e-12) - 20.0 * np.log10(np.min(np.abs(rx_frame[rx_frame != 0])) + 1e-12))

        metric = self._compute_metric(snr, motion_score, dynamic_range)
        self._metric_history.append(metric)

        n_before_update = 5
        if len(self._snr_history) < n_before_update:
            return AdaptationResult(
                spread_hz=current_config.chaotic_spread_hz,
                chaotic_rate=current_config.chaotic_rate,
                amplitude=current_config.tone_amplitude,
                snr_estimate=snr,
                metric_history=tuple(self._metric_history[-64:]),
            )

        avg_snr = float(np.mean(self._snr_history))
        if metric > self._best_metric:
            self._best_metric = metric
            self._best_params = (self._spread, self._rate, self._amplitude)

        lr = self.config.learning_rate
        param_idx = self._step_index % 3

        if param_idx == 0:
            delta = lr * (self.config.spread_hz_max - self.config.spread_hz_min)
            self._spread += delta * (1.0 if avg_snr < 15.0 else -1.0)
            self._spread = float(np.clip(self._spread, self.config.spread_hz_min, self.config.spread_hz_max))
        elif param_idx == 1:
            delta = lr * (self.config.chaotic_rate_max - self.config.chaotic_rate_min)
            self._rate += delta * (1.0 if motion_score < 0.001 else -1.0)
            self._rate = float(np.clip(self._rate, self.config.chaotic_rate_min, self.config.chaotic_rate_max))
        else:
            delta = lr * (self.config.amplitude_max - self.config.amplitude_min)
            self._amplitude += delta * (1.0 if snr < 10.0 else -0.5)
            self._amplitude = float(np.clip(self._amplitude, self.config.amplitude_min, self.config.amplitude_max))

        self._step_index += 1
        metric_tuple = tuple(self._metric_history[-64:])

        return AdaptationResult(
            spread_hz=self._spread,
            chaotic_rate=self._rate,
            amplitude=self._amplitude,
            snr_estimate=avg_snr,
            metric_history=metric_tuple,
        )

    def reset(self) -> None:
        self._snr_history.clear()
        self._metric_history.clear()
        self._spread = self.config.spread_hz_min
        self._rate = self.config.chaotic_rate_min
        self._amplitude = self.config.amplitude_min
        self._step_index = 0
        self._best_metric = -1e9
