from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config.presets import RadioConfig
from .backend import BackendUnavailableError, IQFrame, RadioBackend


_DTYPE_COMPLEX64 = np.dtype(np.complex64)
_DTYPE_INT8 = np.dtype(np.int8)


def _try_import_python_hackrf():
    try:
        import hackrf as _hackrf
        return _hackrf
    except ImportError:
        return None


def _try_import_soapysdr():
    try:
        import SoapySDR as _soapy
        return _soapy
    except ImportError:
        return None


@dataclass(slots=True)
class _PythonHackrfDevice:
    index: int
    serial: str
    usb_version: str


class _PythonHackrfWrapper:
    """Wrapper around native python_hackrf for half-duplex TX/RX."""

    def __init__(self, radio_config: RadioConfig) -> None:
        self._hackrf = _try_import_python_hackrf()
        if self._hackrf is None:
            raise BackendUnavailableError("python_hackrf is not installed.")
        self._radio_config = radio_config
        self._dev = None
        self._opened = False

    def open(self) -> None:
        if self._opened:
            return
        index = self._radio_config.device_index
        self._dev = self._hackrf.HackRF(device_index=index)
        self._dev.sample_rate = int(self._radio_config.sample_rate_sps)
        self._dev.center_freq = int(self._radio_config.center_frequency_hz)
        self._dev.tx_gain = int(self._radio_config.tx_gain_db)
        self._dev.rx_gain = int(self._radio_config.rx_gain_db)
        try:
            self._dev.amplifier_on = True
        except AttributeError:
            pass
        self._opened = True

    def close(self) -> None:
        if self._dev is not None:
            try:
                self._dev.stop_tx()
            except Exception:
                pass
            try:
                self._dev.stop_rx()
            except Exception:
                pass
            try:
                self._dev.close()
            except Exception:
                pass
        self._dev = None
        self._opened = False

    @staticmethod
    def _complex_to_int8(samples: np.ndarray) -> bytes:
        clipped = np.clip(np.rint(samples.view(np.float32).reshape(-1, 2)), -128.0, 127.0)
        return clipped.astype(np.int8).tobytes()

    @staticmethod
    def _int8_to_complex(buffer: bytes) -> np.ndarray:
        raw = np.frombuffer(buffer, dtype=np.int8).astype(np.float32)
        N = raw.size // 2
        return (raw[: 2 * N : 2] + 1j * raw[1 : 2 * N : 2]).astype(np.complex64) / 128.0

    def transmit(self, samples: np.ndarray) -> int:
        if not self._opened or self._dev is None:
            raise BackendUnavailableError("python_hackrf device not open")

        samples = np.asarray(samples, dtype=np.complex64)
        tx_bytes = self._complex_to_int8(samples)
        tx_len = len(tx_bytes)
        tx_sent = [0]

        def _tx_callback(buffer_out: bytearray) -> int:
            n = min(len(buffer_out), tx_len - tx_sent[0])
            buffer_out[:n] = tx_bytes[tx_sent[0] : tx_sent[0] + n]
            tx_sent[0] += n
            tx_len_remaining = tx_len - tx_sent[0]
            if tx_len_remaining <= 0:
                return -1
            if tx_len_remaining < len(buffer_out):
                return tx_len_remaining
            return len(buffer_out)

        self._dev.start_tx(_tx_callback)
        while tx_sent[0] < tx_len:
            time.sleep(0.0001)
        self._dev.stop_tx()
        return int(samples.size)

    def receive(self, num_samples: int, _time_step_s: float | None = None) -> IQFrame:
        if not self._opened or self._dev is None:
            raise BackendUnavailableError("python_hackrf device not open")

        rx_buf = bytearray()
        target_bytes = num_samples * 2  # 2 bytes per sample (I8+Q8)

        def _rx_callback(buffer_in: bytes) -> int:
            nonlocal rx_buf
            rx_buf.extend(buffer_in)
            return 0

        self._dev.start_rx(_rx_callback)
        while len(rx_buf) < target_bytes:
            time.sleep(0.0001)
        self._dev.stop_rx()

        iq = self._int8_to_complex(bytes(rx_buf[:target_bytes]))
        return IQFrame(
            samples=iq.astype(np.complex64, copy=False),
            timestamp_s=time.monotonic(),
            backend_name="hackrf(python_hackrf)",
            center_frequency_hz=self._radio_config.center_frequency_hz,
            sample_rate_sps=self._radio_config.sample_rate_sps,
            metadata={"device_index": float(self._radio_config.device_index)},
        )


class _SoapySdrWrapper:
    """SoapySDR-based HackRF adapter (fallback)."""

    def __init__(self, radio_config: RadioConfig) -> None:
        self._soapysdr = _try_import_soapysdr()
        if self._soapysdr is None:
            raise BackendUnavailableError("SoapySDR is not installed.")
        self._radio_config = radio_config
        self._sdr = None
        self._rx_stream = None
        self._tx_stream = None
        self._opened = False

    def open(self) -> None:
        if self._opened:
            return
        SoapySDR = self._soapysdr
        self._sdr = SoapySDR.Device({"driver": "hackrf", "label": str(self._radio_config.device_index)})
        self._sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self._radio_config.sample_rate_sps)
        self._sdr.setSampleRate(SoapySDR.SOAPY_SDR_TX, 0, self._radio_config.sample_rate_sps)
        self._sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, self._radio_config.center_frequency_hz)
        self._sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, self._radio_config.center_frequency_hz)
        self._sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, self._radio_config.rx_gain_db)
        self._sdr.setGain(SoapySDR.SOAPY_SDR_TX, 0, self._radio_config.tx_gain_db)
        self._rx_stream = self._sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        self._tx_stream = self._sdr.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32)
        self._sdr.activateStream(self._rx_stream)
        self._sdr.activateStream(self._tx_stream)
        self._opened = True

    def close(self) -> None:
        if self._sdr is None:
            self._opened = False
            return
        SoapySDR = self._soapysdr
        try:
            if self._rx_stream is not None:
                self._sdr.deactivateStream(self._rx_stream)
                self._sdr.closeStream(self._rx_stream)
            if self._tx_stream is not None:
                self._sdr.deactivateStream(self._tx_stream)
                self._sdr.closeStream(self._tx_stream)
        finally:
            self._opened = False
            self._rx_stream = None
            self._tx_stream = None
            self._sdr = None

    def transmit(self, samples: np.ndarray) -> int:
        if not self._opened or self._sdr is None:
            raise BackendUnavailableError("SoapySDR backend not open")
        vector = np.asarray(samples, dtype=np.complex64)
        result = self._sdr.writeStream(self._tx_stream, [vector], vector.size, timeoutUs=100000)
        return int(getattr(result, "ret", result))

    def receive(self, num_samples: int, _time_step_s: float | None = None) -> IQFrame:
        if not self._opened or self._sdr is None:
            raise BackendUnavailableError("SoapySDR backend not open")
        buffer = np.empty(num_samples, dtype=np.complex64)
        result = self._sdr.readStream(self._rx_stream, [buffer], num_samples, timeoutUs=100000)
        received = int(getattr(result, "ret", result))
        if received <= 0:
            buffer.fill(0.0)
            received = num_samples
        return IQFrame(
            samples=buffer[:received],
            timestamp_s=time.monotonic(),
            backend_name="hackrf(soapysdr)",
            center_frequency_hz=self._radio_config.center_frequency_hz,
            sample_rate_sps=self._radio_config.sample_rate_sps,
            metadata={"device_index": float(self._radio_config.device_index)},
        )


class HackRFBackend(RadioBackend):
    """HackRF backend with auto-detection: python_hackrf -> SoapySDR."""

    name = "hackrf"

    def __init__(self, radio_config: RadioConfig) -> None:
        self.radio_config = radio_config
        self._driver: _PythonHackrfWrapper | _SoapySdrWrapper | None = None
        self._opened = False

    @property
    def active_driver_name(self) -> str:
        if isinstance(self._driver, _PythonHackrfWrapper):
            return "python_hackrf"
        if isinstance(self._driver, _SoapySdrWrapper):
            return "soapysdr"
        return "none"

    def open(self) -> None:
        if self._opened:
            return

        try:
            self._driver = _PythonHackrfWrapper(self.radio_config)
            self._driver.open()
            self._opened = True
            return
        except (BackendUnavailableError, ImportError, OSError):
            pass

        try:
            self._driver = _SoapySdrWrapper(self.radio_config)
            self._driver.open()
            self._opened = True
            return
        except (BackendUnavailableError, ImportError, OSError) as exc:
            raise BackendUnavailableError(
                f"No HackRF driver available. Tried python_hackrf and SoapySDR. Error: {exc}"
            ) from exc

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
        self._driver = None
        self._opened = False

    def transmit(self, samples: np.ndarray) -> int:
        if self._driver is None:
            raise BackendUnavailableError("HackRF backend not open")
        return self._driver.transmit(samples)

    def receive(self, num_samples: int, time_step_s: float | None = None) -> IQFrame:
        if self._driver is None:
            raise BackendUnavailableError("HackRF backend not open")
        return self._driver.receive(num_samples, time_step_s)
