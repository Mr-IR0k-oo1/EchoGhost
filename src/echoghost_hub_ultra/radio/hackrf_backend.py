"""HackRF backend adapter.

The default implementation uses SoapySDR because it is the most widely
available Python-accessible path to HackRF on Windows and Linux.
"""

from __future__ import annotations

import time

import numpy as np

from ..config.presets import RadioConfig
from .backend import BackendUnavailableError, IQFrame, RadioBackend


class HackRFBackend(RadioBackend):
    """Minimal HackRF adapter built on top of SoapySDR."""

    name = "hackrf"

    def __init__(self, radio_config: RadioConfig) -> None:
        self.radio_config = radio_config
        self._sdr = None
        self._soapysdr = None
        self._rx_stream = None
        self._tx_stream = None
        self._opened = False

    def open(self) -> None:
        try:
            import SoapySDR  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on local hardware setup
            raise BackendUnavailableError(
                "SoapySDR is not installed. Install SoapySDR and the HackRF driver to use hardware mode."
            ) from exc

        self._soapysdr = SoapySDR
        self._sdr = SoapySDR.Device({"driver": "hackrf", "label": str(self.radio_config.device_index)})
        self._sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self.radio_config.sample_rate_sps)
        self._sdr.setSampleRate(SoapySDR.SOAPY_SDR_TX, 0, self.radio_config.sample_rate_sps)
        self._sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, self.radio_config.center_frequency_hz)
        self._sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, self.radio_config.center_frequency_hz)
        self._sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, self.radio_config.rx_gain_db)
        self._sdr.setGain(SoapySDR.SOAPY_SDR_TX, 0, self.radio_config.tx_gain_db)
        self._rx_stream = self._sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        self._tx_stream = self._sdr.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32)
        self._sdr.activateStream(self._rx_stream)
        self._sdr.activateStream(self._tx_stream)
        self._opened = True

    def close(self) -> None:
        if self._sdr is None or self._soapysdr is None:
            self._opened = False
            return

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
            self._soapysdr = None

    def transmit(self, samples: np.ndarray) -> int:
        if not self._opened or self._sdr is None or self._soapysdr is None or self._tx_stream is None:
            raise BackendUnavailableError("HackRF backend is not open.")

        vector = np.asarray(samples, dtype=np.complex64)
        result = self._sdr.writeStream(self._tx_stream, [vector], vector.size, timeoutUs=100000)
        return int(getattr(result, "ret", result))

    def receive(self, num_samples: int, time_step_s: float | None = None) -> IQFrame:
        if not self._opened or self._sdr is None or self._soapysdr is None or self._rx_stream is None:
            raise BackendUnavailableError("HackRF backend is not open.")

        buffer = np.empty(num_samples, dtype=np.complex64)
        result = self._sdr.readStream(self._rx_stream, [buffer], num_samples, timeoutUs=100000)
        received = int(getattr(result, "ret", result))
        if received <= 0:
            buffer.fill(0.0)
            received = num_samples
        timestamp = time.monotonic()
        return IQFrame(
            samples=buffer[:received],
            timestamp_s=timestamp,
            backend_name=self.name,
            center_frequency_hz=self.radio_config.center_frequency_hz,
            sample_rate_sps=self.radio_config.sample_rate_sps,
            metadata={"device_index": float(self.radio_config.device_index)},
        )

