from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Thread

import numpy as np


@dataclass(slots=True)
class ArtState:
    motion_score: float = 0.0
    breathing_bpm: float = 15.0
    spectrum_energy: float = 0.0
    doppler_spread: float = 0.0
    activity_label: str = "idle"


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class ArtGenerator:
    """Generative art engine that maps RF features to sound and visuals.

    Sound:
      - Maps motion score -> note velocity
      - Maps breathing BPM -> LFO rate on a tone
      - Maps Doppler spread -> filter cutoff / timbre brightness

    Visuals:
      - Provides RGB colour palette from RF features
      - Generates particle-like animation parameters
    """

    def __init__(self, sample_rate_hz: int = 44100) -> None:
        self.sample_rate = sample_rate_hz
        self._state = ArtState()
        self._phase: float = 0.0
        self._audio_buffer: deque[float] = deque(maxlen=sample_rate_hz * 2)
        self._running = False
        self._audio_thread: Thread | None = None
        self._visual_history: deque[dict[str, float]] = deque(maxlen=128)

    def update(self, state: ArtState) -> None:
        self._state = state

    @property
    def state(self) -> ArtState:
        return self._state

    def _map_to_note(self, value: float, min_val: float, max_val: float, octave: int = 4) -> tuple[int, float]:
        normalised = (value - min_val) / (max_val - min_val + 1e-12)
        normalised = float(np.clip(normalised, 0.0, 1.0))
        note_index = int(normalised * 11.0)
        midi = (octave + 1) * 12 + note_index
        velocity = int(20 + 100 * normalised)
        return midi, velocity / 127.0

    def generate_sound_samples(self, num_samples: int) -> np.ndarray:
        s = self._state
        motion = float(np.clip(s.motion_score * 500.0, 0.0, 1.0))
        bpm = float(max(s.breathing_bpm, 4.0))
        lfo_hz = bpm / 60.0
        carrier_hz = 110.0 + 440.0 * motion
        spread = float(np.clip(s.doppler_spread * 2.0, 0.0, 1.0))

        t = np.arange(num_samples, dtype=np.float64) / float(self.sample_rate)
        full_t = self._phase + t
        lfo = 0.5 + 0.5 * np.sin(math.tau * lfo_hz * full_t)
        carrier = np.sin(math.tau * carrier_hz * full_t + spread * np.sin(math.tau * 3.0 * full_t))
        envelope = lfo * (0.3 + 0.7 * motion)
        samples = 0.15 * envelope * carrier
        self._phase = float((full_t[-1] + 1.0 / self.sample_rate) % (1.0 / lfo_hz if lfo_hz > 0 else 1.0))

        return samples.astype(np.float32, copy=False)

    def start_audio(self) -> None:
        if self._running:
            return
        self._running = True
        self._audio_thread = Thread(target=self._audio_loop, name="art-audio", daemon=True)
        self._audio_thread.start()

    def stop_audio(self) -> None:
        self._running = False
        if self._audio_thread is not None:
            self._audio_thread.join(timeout=1.0)
        self._audio_thread = None

    def _audio_loop(self) -> None:
        try:
            import sounddevice as sd
            chunk = 512
            while self._running:
                samples = self.generate_sound_samples(chunk)
                sd.play(samples, samplerate=self.sample_rate, blocking=True)
        except ImportError:
            pass

    def colour_palette(self) -> tuple[float, float, float]:
        s = self._state
        r = float(np.clip(0.2 + 0.8 * s.motion_score * 300.0, 0.0, 1.0))
        g = float(np.clip(0.1 + 0.6 * s.doppler_spread, 0.0, 1.0))
        b = float(np.clip(0.8 - 0.6 * s.motion_score * 200.0, 0.0, 1.0))
        return (r, g, b)

    def visual_params(self) -> dict[str, float]:
        s = self._state
        params = {
            "particle_size": float(np.clip(2.0 + 20.0 * s.motion_score * 200.0, 2.0, 22.0)),
            "rotation_speed": float(np.clip(s.breathing_bpm / 60.0, 0.1, 2.0)),
            "wave_amplitude": float(np.clip(s.doppler_spread * 50.0, 0.0, 50.0)),
            "opacity": float(np.clip(0.3 + 0.7 * s.motion_score * 300.0, 0.3, 1.0)),
            "trail_length": int(np.clip(5 + int(s.motion_score * 2000.0), 5, 50)),
        }
        self._visual_history.append(params)
        return params

    def reset(self) -> None:
        self._state = ArtState()
        self._phase = 0.0
        self._audio_buffer.clear()
        self._visual_history.clear()
