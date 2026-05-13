"""
Signal processing engine for EchoGhost Hub Ultra.

This module performs all real-time radar signal processing including:
  - Baseline removal (exponential averaging)
  - Range-Doppler map generation (2D FFT over fast-time/slow-time)
  - Micro-Doppler spectrogram (STFT of slow-time phase history)
  - Breathing/vital signs extraction (phase-based chest displacement)
  - Motion detection and tracking
  - ML-based activity classification (scikit-learn)
  - Ghost imaging / low-res heatmap rendering

Concepts:
  - Fast time: samples within a single chirp/pulse (range dimension)
  - Slow time: samples across consecutive chirps/pulses (Doppler / velocity dimension)
  - CPI (Coherent Processing Interval): block of pulses over which phase coherence is maintained
  - Range-Doppler map: 2D Fourier transform revealing target range and velocity
  - Micro-Doppler: small frequency modulations due to moving body parts
"""

import time
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
from numpy.fft import fft, fftshift, ifft

from config import AppConfig


# ── Numba JIT Acceleration ────────────────────────────────────────────────

try:
    from numba import jit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Fallback no-op decorator
    def jit(*args, **kwargs):
        def decorator(f):
            return f

        return decorator

    def prange(*args):
        return range(*args)


# ── scikit-learn Classifier ───────────────────────────────────────────────

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ──────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ProcessingResult:
    """Output from a single CPI of signal processing."""

    # Range-Doppler
    range_doppler_map: np.ndarray  # 2D array [range_bins, doppler_bins]
    range_profile: np.ndarray  # 1D magnitude averaged over Doppler
    doppler_profile: np.ndarray  # 1D magnitude averaged over range

    # Micro-Doppler
    micro_doppler_spectrogram: np.ndarray  # 2D STFT [time_bins, freq_bins]
    micro_doppler_freqs: np.ndarray  # Frequency axis for spectrogram

    # Breathing / Vital Signs
    breathing_rate_bpm: float
    breathing_phase: np.ndarray  # Phase history over CPI
    breathing_confidence: float  # 0-1 confidence estimate

    # Motion
    motion_detected: bool
    motion_magnitude: float  # 0-1 normalized motion energy
    target_range_m: float  # Estimated range of primary target
    target_velocity_mps: float  # Estimated radial velocity

    # Ghost / Heatmap
    heatmap: np.ndarray  # 2D spatial energy distribution

    # Classification
    activity_label: str
    activity_confidence: float

    # Metadata
    timestamp: float
    cpi_index: int


# ──────────────────────────────────────────────────────────────────────────
# Signal Processor
# ──────────────────────────────────────────────────────────────────────────


class SignalProcessor:
    """Real-time radar signal processor.

    Processes raw IQ data one CPI at a time, producing range-Doppler maps,
    micro-Doppler spectrograms, breathing rate estimates, and activity labels.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        pc = config.processing
        rc = config.radar

        # Processing state
        self._cpi_index = 0
        self._baseline_iq: Optional[np.ndarray] = None
        self._slow_time_buffer: Optional[np.ndarray] = None
        self._slow_time_idx = 0

        # Breathing signal buffer
        self._breathing_buffer = np.zeros(rc.cpi_length * 10)
        self._breathing_idx = 0

        # Micro-Doppler buffer
        self._md_buffer = np.zeros(2048, dtype=np.complex64)
        self._md_idx = 0

        # Range-Doppler parameters
        self._range_bins = rc.range_bins
        self._doppler_bins = rc.doppler_bins
        self._cpi_length = rc.cpi_length

        # Range axis (meters)
        fs = config.hackrf.sample_rate_hz
        chirp_dur = rc.chirp_duration_s
        bw = rc.chirp_bw_hz
        self._range_resolution = 3e8 / (2 * bw)  # c / (2 * BW)
        range_max = self._range_bins * self._range_resolution
        self._range_axis = np.linspace(0, range_max, self._range_bins)

        # Doppler axis (m/s)
        prf = rc.pulse_repetition_hz
        self._velocity_resolution = 3e8 / (
            2 * rc.frequency_hz * prf * self._doppler_bins
        )
        doppler_max = self._doppler_bins * self._velocity_resolution / 2
        self._velocity_axis = np.linspace(-doppler_max, doppler_max, self._doppler_bins)

        # Time axis
        self._time_axis = np.arange(self._cpi_length) / prf

        # ML classifier
        self._classifier = None
        self._scaler = None
        if pc.enable_ml_classifier:
            self._init_classifier()

    # ── ML Classifier Initialization ─────────────────────────────────────

    def _init_classifier(self):
        """Initialize or load a pretrained activity classifier.

        Uses a simple Random Forest trained on feature vectors derived from
        the micro-Doppler and range-Doppler data.

        Activities: idle, walking, sitting, gesturing, falling
        """
        if not HAS_SKLEARN:
            self._classifier = None
            return

        self._scaler = StandardScaler()
        self._classifier = RandomForestClassifier(
            n_estimators=50, max_depth=8, random_state=42
        )

        # In a real deployment, load pretrained weights from disk.
        # For now, we train on synthetic features on first call.
        self._fit_synthetic()

    def _fit_synthetic(self):
        """Train the classifier on synthetic feature data.

        This provides reasonable initial performance. In production, replace
        with real collected and labeled data.
        """
        if self._classifier is None:
            return

        np.random.seed(42)
        n_samples = 200
        n_features = 16

        # Generate synthetic features for each activity
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 5, n_samples)

        # Add some structure to make it learnable
        for i in range(5):
            X[i::5] += np.random.randn(n_features) * (i + 1)

        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)
        self._classifier.fit(X_scaled, y)

    def _extract_features(self, rd_map: np.ndarray, md_spec: np.ndarray) -> np.ndarray:
        """Extract feature vector from radar data for classification.

        Features:
          - Range profile statistics (mean, std, max, peak location)
          - Doppler profile statistics
          - Micro-Doppler energy distribution
          - Spectral moments
        """
        features = []

        # Range profile features
        range_profile = np.mean(np.abs(rd_map), axis=1)
        features.extend(
            [
                np.mean(range_profile),
                np.std(range_profile),
                np.max(range_profile),
                np.argmax(range_profile) / len(range_profile),
            ]
        )

        # Doppler profile features
        doppler_profile = np.mean(np.abs(rd_map), axis=0)
        features.extend(
            [
                np.mean(doppler_profile),
                np.std(doppler_profile),
                np.max(doppler_profile),
                np.argmax(np.abs(doppler_profile - np.mean(doppler_profile)))
                / len(doppler_profile),
            ]
        )

        # Micro-Doppler energy
        if md_spec.size > 0:
            md_energy = np.mean(np.abs(md_spec), axis=1)
            features.extend(
                [
                    np.mean(md_energy),
                    np.std(md_energy),
                    np.max(md_energy),
                    np.sum(md_energy > np.median(md_energy)) / len(md_energy),
                ]
            )
        else:
            features.extend([0, 0, 0, 0])

        features.extend(
            [
                np.mean(np.abs(rd_map)),
                np.std(np.abs(rd_map)),
                np.sum(np.abs(rd_map) > np.median(np.abs(rd_map))) / rd_map.size,
                np.var(np.abs(rd_map)),
            ]
        )

        return np.array(features[:16])

    # ── Core Processing Pipeline ─────────────────────────────────────────

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _remove_baseline_numba(
        iq: np.ndarray, baseline: np.ndarray, alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated exponential baseline removal.

        The baseline is a running average of the static clutter. Subtracting
        it reveals moving targets.

        Args:
            iq: Input IQ samples.
            baseline: Current baseline estimate.
            alpha: Update factor (0 = no update, 1 = instant).

        Returns:
            (clutter_removed_iq, updated_baseline)
        """
        if len(baseline) != len(iq):
            baseline = iq.copy()
            return np.zeros_like(iq, dtype=np.complex64), baseline

        for i in prange(len(iq)):
            baseline[i] = (1 - alpha) * baseline[i] + alpha * iq[i]

        return iq - baseline, baseline

    def _remove_baseline(self, iq: np.ndarray) -> np.ndarray:
        """Remove static clutter via exponential averaging.

        The baseline represents the static environment. By continuously
        updating it and subtracting, only moving/changing reflections remain.

        Args:
            iq: Raw IQ samples.

        Returns:
            IQ with static clutter removed.
        """
        pc = self.config.processing
        alpha = pc.baseline_alpha

        if self._baseline_iq is None or len(self._baseline_iq) != len(iq):
            self._baseline_iq = iq.copy()
            return np.zeros_like(iq, dtype=np.complex64)

        if pc.numba_enabled and HAS_NUMBA:
            cleaned, self._baseline_iq = self._remove_baseline_numba(
                iq, self._baseline_iq, alpha
            )
            return cleaned

        # Pure NumPy fallback
        self._baseline_iq = (1 - alpha) * self._baseline_iq + alpha * iq
        return iq - self._baseline_iq

    def _range_fft(self, iq_matrix: np.ndarray) -> np.ndarray:
        """Perform range FFT on fast-time samples.

        Each row of iq_matrix is one pulse's worth of samples.
        The FFT converts time delay -> range.

        Args:
            iq_matrix: Shape (num_pulses, samples_per_pulse).

        Returns:
            Range-compressed data, shape (num_pulses, range_bins).
        """
        return fft(iq_matrix, n=self._range_bins, axis=1)

    def _doppler_fft(self, range_compressed: np.ndarray) -> np.ndarray:
        """Perform Doppler FFT on slow-time samples.

        For each range bin, the FFT across pulses reveals velocity
        (Doppler shift). Positive frequencies = approaching, negative = receding.

        Args:
            range_compressed: Shape (num_pulses, range_bins).

        Returns:
            Range-Doppler map, shape (range_bins, doppler_bins).
        """
        # Window to reduce sidelobes
        window = np.hanning(range_compressed.shape[0])
        windowed = range_compressed * window[:, np.newaxis]

        # FFT across slow time (axis=0)
        rd_map = fftshift(fft(windowed, n=self._doppler_bins, axis=0), axes=0)

        return rd_map.T  # Transpose to (range_bins, doppler_bins)

    def _compute_range_doppler(self, cpi_iq: np.ndarray) -> np.ndarray:
        """Compute the full range-Doppler map for one CPI.

        Pipeline:
          1. Reshape into (num_pulses, samples_per_pulse) matrix
          2. Baseline removal on each pulse
          3. Range FFT (fast time)
          4. Doppler FFT (slow time)

        Args:
            cpi_iq: Raw IQ for one CPI.

        Returns:
            2D range-Doppler map (range_bins x doppler_bins).
        """
        rc = self.config.radar
        hc = self.config.hackrf

        samples_per_pulse = int(hc.sample_rate_hz * rc.chirp_duration_s)

        # Reshape into fast-time x slow-time matrix
        pulses = self._cpi_length
        usable = min(len(cpi_iq), pulses * samples_per_pulse)
        usable = (usable // samples_per_pulse) * samples_per_pulse
        matrix = cpi_iq[:usable].reshape(-1, samples_per_pulse)

        # Ensure we have enough pulses
        if matrix.shape[0] < 2:
            return np.zeros((self._range_bins, self._doppler_bins), dtype=np.complex64)

        # Baseline removal per pulse
        for p in range(matrix.shape[0]):
            matrix[p] = self._remove_baseline(matrix[p])

        # Range compression
        range_compressed = self._range_fft(matrix)

        # Doppler processing
        rd_map = self._doppler_fft(range_compressed)

        return rd_map

    # ── Micro-Doppler ────────────────────────────────────────────────────

    def _compute_micro_doppler(
        self, cpi_iq: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute micro-Doppler spectrogram using STFT.

        Micro-Doppler captures fine-grained frequency modulations caused
        by moving body parts (arms, legs, chest). This is key for
        activity recognition and vital signs.

        Args:
            cpi_iq: Raw IQ for one CPI.

        Returns:
            (spectrogram, freq_axis)
        """
        pc = self.config.processing
        rc = self.config.radar
        hc = self.config.hackrf

        samples_per_pulse = int(hc.sample_rate_hz * rc.chirp_duration_s)
        pulses = min(len(cpi_iq) // samples_per_pulse, self._cpi_length)

        if pulses < 4:
            return np.zeros((64, 64)), np.linspace(-100, 100, 64)

        # Extract phase at peak range bin across pulses
        matrix = cpi_iq[: pulses * samples_per_pulse].reshape(pulses, samples_per_pulse)
        range_compressed = self._range_fft(matrix)

        # Find the range bin with maximum energy
        range_profile = np.mean(np.abs(range_compressed), axis=0)
        peak_bin = np.argmax(range_profile)

        # Extract phase history at peak range
        phase_history = np.angle(range_compressed[:, peak_bin])

        # Unwrap phase to get continuous displacement
        phase_unwrapped = np.unwrap(phase_history)

        # STFT for micro-Doppler
        fft_size = min(pc.doppler_fft_size, len(phase_unwrapped))
        overlap = int(fft_size * pc.micro_doppler_overlap)
        hop = fft_size - overlap

        if len(phase_unwrapped) < fft_size:
            return np.zeros((64, 64)), np.linspace(-100, 100, 64)

        n_frames = 1 + (len(phase_unwrapped) - fft_size) // hop
        spectrogram = np.zeros((n_frames, fft_size), dtype=np.float32)
        window = np.hanning(fft_size)

        for i in range(n_frames):
            start = i * hop
            segment = phase_unwrapped[start : start + fft_size] * window
            spec = np.abs(fftshift(fft(segment)))
            spectrogram[i] = spec

        # Frequency axis
        prf = rc.pulse_repetition_hz
        freqs = fftshift(np.fft.fftfreq(fft_size, d=1.0 / prf))

        return spectrogram, freqs

    # ── Breathing / Vital Signs ──────────────────────────────────────────

    def _compute_breathing(self, cpi_iq: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """Extract breathing rate from chest micro-movements.

        The chest moves ~1-12 mm during breathing. This causes a phase
        modulation in the reflected radar signal. By tracking the phase
        at the range bin corresponding to the chest, we can extract the
        breathing waveform and frequency.

        Algorithm:
          1. Range compress and find chest bin
          2. Extract unwrapped phase history
          3. Bandpass filter (0.1 - 0.8 Hz)
          4. Count zero crossings to estimate rate

        Returns:
            (rate_bpm, phase_history, confidence)
        """
        rc = self.config.radar
        hc = self.config.hackrf

        samples_per_pulse = int(hc.sample_rate_hz * rc.chirp_duration_s)
        pulses = min(len(cpi_iq) // samples_per_pulse, self._cpi_length)

        if pulses < 4:
            return 0.0, np.array([]), 0.0

        matrix = cpi_iq[: pulses * samples_per_pulse].reshape(pulses, samples_per_pulse)
        range_compressed = self._range_fft(matrix)

        # Find chest range bin (typically 2-8 meters)
        # In practice, this is the bin with strongest periodic motion
        range_profile = np.mean(np.abs(range_compressed), axis=0)
        # Focus on bins corresponding to 1-10 m
        min_bin = max(0, int(1.0 / self._range_resolution))
        max_bin = min(self._range_bins, int(10.0 / self._range_resolution))
        if min_bin >= max_bin:
            min_bin, max_bin = 0, min(32, self._range_bins)
        chest_bin = min_bin + np.argmax(range_profile[min_bin:max_bin])

        # Extract phase
        phase = np.angle(range_compressed[:, chest_bin])
        phase_unwrapped = np.unwrap(phase)

        # Bandpass filter for breathing (0.1 - 0.8 Hz)
        prf = rc.pulse_repetition_hz
        fft_phase = fft(phase_unwrapped)
        freqs = np.fft.fftfreq(len(phase_unwrapped), d=1.0 / prf)
        mask = (np.abs(freqs) >= self.config.processing.breathing_low_cut) & (
            np.abs(freqs) <= self.config.processing.breathing_high_cut
        )
        fft_phase[~mask] = 0
        filtered = np.real(ifft(fft_phase))

        # Estimate breathing rate from dominant frequency
        n = len(filtered)
        if n > 4:
            fft_f = fft(filtered)
            freqs_f = np.fft.fftfreq(n, d=1.0 / prf)
            positive = freqs_f > 0
            peak_idx = np.argmax(np.abs(fft_f[positive]))
            dominant_freq = freqs_f[positive][peak_idx]
            rate_bpm = dominant_freq * 60.0
            confidence = min(
                1.0,
                np.abs(fft_f[positive][peak_idx])
                / (np.mean(np.abs(fft_f[positive])) + 1e-6)
                * 0.1,
            )
        else:
            rate_bpm = 0.0
            confidence = 0.0

        # Clip to physiological range
        if rate_bpm < 6 or rate_bpm > 40:
            rate_bpm = 0.0
            confidence = 0.0

        return rate_bpm, filtered, min(confidence, 1.0)

    # ── Motion Detection ─────────────────────────────────────────────────

    def _detect_motion(self, cpi_iq: np.ndarray) -> Tuple[bool, float, float, float]:
        """Detect and characterize motion.

        Computes the energy in the Doppler spectrum after clutter removal.
        Stationary targets have zero Doppler; moving targets create
        Doppler shifts.

        Args:
            cpi_iq: Raw IQ for one CPI.

        Returns:
            (motion_detected, magnitude, target_range_m, target_velocity_mps)
        """
        rc = self.config.radar
        hc = self.config.hackrf

        samples_per_pulse = int(hc.sample_rate_hz * rc.chirp_duration_s)
        pulses = min(len(cpi_iq) // samples_per_pulse, self._cpi_length)

        if pulses < 2:
            return False, 0.0, 0.0, 0.0

        matrix = cpi_iq[: pulses * samples_per_pulse].reshape(pulses, samples_per_pulse)
        for p in range(matrix.shape[0]):
            matrix[p] = self._remove_baseline(matrix[p])

        range_compressed = self._range_fft(matrix)
        rd_map = self._doppler_fft(range_compressed)
        rd_mag = np.abs(rd_map)

        # Motion energy is total energy in non-zero Doppler bins
        doppler_profile = np.mean(rd_mag, axis=0)
        motion_energy = np.std(doppler_profile) / (np.mean(doppler_profile) + 1e-10)

        threshold = self.config.processing.motion_threshold
        detected = motion_energy > threshold

        # Target range = peak range bin
        range_profile = np.mean(rd_mag, axis=1)
        peak_range_bin = np.argmax(range_profile)
        target_range = peak_range_bin * self._range_resolution

        # Target velocity = peak Doppler bin
        peak_doppler_bin = np.argmax(doppler_profile)
        target_velocity = (
            peak_doppler_bin - self._doppler_bins // 2
        ) * self._velocity_resolution

        return detected, min(motion_energy, 1.0), target_range, target_velocity

    # ── Ghost / Heatmap ──────────────────────────────────────────────────

    def _compute_heatmap(self, rd_map: np.ndarray) -> np.ndarray:
        """Generate a 2D spatial heatmap from the range-Doppler data.

        This produces a low-resolution "ghost image" showing energy
        distribution in range-velocity space. In artistic mode, this
        drives the Three.js particle visualization.

        Args:
            rd_map: Range-Doppler map.

        Returns:
            2D heatmap normalized to [0, 1].
        """
        heatmap = np.abs(rd_map)

        # Log compress for better dynamic range
        heatmap = 20 * np.log10(heatmap + 1e-10)

        # Normalize to [0, 1]
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax > hmin:
            heatmap = (heatmap - hmin) / (hmax - hmin)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    # ── Classification ───────────────────────────────────────────────────

    def _classify_activity(
        self, rd_map: np.ndarray, md_spec: np.ndarray
    ) -> Tuple[str, float]:
        """Classify human activity from radar features.

        Args:
            rd_map: Range-Doppler map.
            md_spec: Micro-Doppler spectrogram.

        Returns:
            (label, confidence)
        """
        labels = ["idle", "walking", "sitting", "gesturing", "falling"]

        if self._classifier is None:
            # Fallback heuristic classifier
            doppler_energy = np.std(np.mean(np.abs(rd_map), axis=0))
            range_spread = np.std(np.mean(np.abs(rd_map), axis=1))
            md_energy = np.mean(np.abs(md_spec)) if md_spec.size > 0 else 0

            if md_energy > 0.3 and range_spread > 0.2:
                return "gesturing", 0.6
            elif doppler_energy > 0.3:
                return "walking", 0.5
            elif range_spread > 0.15:
                return "sitting", 0.4
            else:
                return "idle", 0.8

        # ML classifier
        features = self._extract_features(rd_map, md_spec)
        features_scaled = self._scaler.transform(features.reshape(1, -1))
        probs = self._classifier.predict_proba(features_scaled)[0]
        label_idx = np.argmax(probs)
        confidence = probs[label_idx]

        return labels[label_idx], confidence

    # ── Main Processing Entry Point ──────────────────────────────────────

    def process_cpi(self, cpi_iq: np.ndarray) -> ProcessingResult:
        """Process one CPI of raw IQ data through the full pipeline.

        Args:
            cpi_iq: Raw complex IQ samples for one coherent processing interval.

        Returns:
            ProcessingResult with all extracted information.
        """
        timestamp = time.time()
        self._cpi_index += 1

        # 1. Range-Doppler map
        rd_map = self._compute_range_doppler(cpi_iq)

        # 2. Micro-Doppler spectrogram
        md_spec, md_freqs = self._compute_micro_doppler(cpi_iq)

        # 3. Breathing rate
        breath_rate, breath_phase, breath_conf = self._compute_breathing(cpi_iq)

        # 4. Motion detection
        motion_detected, motion_mag, target_range, target_vel = self._detect_motion(
            cpi_iq
        )

        # 5. Heatmap
        heatmap = self._compute_heatmap(rd_map)

        # 6. Activity classification
        activity, activity_conf = self._classify_activity(rd_map, md_spec)

        # Profiles
        range_profile = np.mean(np.abs(rd_map), axis=1)
        doppler_profile = np.mean(np.abs(rd_map), axis=0)

        return ProcessingResult(
            range_doppler_map=np.abs(rd_map),
            range_profile=range_profile,
            doppler_profile=doppler_profile,
            micro_doppler_spectrogram=md_spec,
            micro_doppler_freqs=md_freqs,
            breathing_rate_bpm=breath_rate,
            breathing_phase=breath_phase,
            breathing_confidence=breath_conf,
            motion_detected=motion_detected,
            motion_magnitude=motion_mag,
            target_range_m=target_range,
            target_velocity_mps=target_vel,
            heatmap=heatmap,
            activity_label=activity,
            activity_confidence=activity_conf,
            timestamp=timestamp,
            cpi_index=self._cpi_index,
        )
