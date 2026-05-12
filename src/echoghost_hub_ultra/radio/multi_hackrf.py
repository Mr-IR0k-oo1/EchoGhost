from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config.presets import RadioConfig
from .backend import BackendUnavailableError, IQFrame, RadioBackend
from .hackrf_backend import _PythonHackrfWrapper, _SoapySdrWrapper


@dataclass(slots=True)
class MultiHackrfConfig:
    tx_device_index: int = 0
    rx_device_indices: tuple[int, ...] = (1,)
    combine_rx: bool = True


class MultiHackRFBackend(RadioBackend):
    """Coordinate two or more HackRFs for simultaneous TX/RX.

    Device 0 (default): TX only
    Device(s) 1+: RX only

    This enables true full-duplex operation impossible with a single HackRF.
    """

    name = "multi_hackrf"

    def __init__(self, radio_config: RadioConfig, multi_config: MultiHackrfConfig | None = None) -> None:
        self.radio_config = radio_config
        self.multi_config = multi_config or MultiHackrfConfig()
        self._tx_driver: _PythonHackrfWrapper | _SoapySdrWrapper | None = None
        self._rx_drivers: list[_PythonHackrfWrapper | _SoapySdrWrapper] = []
        self._opened = False

    @staticmethod
    def enumerate_devices() -> list[dict]:
        devices: list[dict] = []
        try:
            import hackrf
            for i in range(4):
                try:
                    d = hackrf.HackRF(device_index=i)
                    devices.append({"index": i, "serial": d.serial_number(), "driver": "python_hackrf"})
                    d.close()
                except Exception:
                    break
        except ImportError:
            pass

        if not devices:
            try:
                import SoapySDR
                results = SoapySDR.Device.enumerate("driver=hackrf")
                for i, info in enumerate(results):
                    devices.append({"index": i, "serial": info.get("serial", f"soapy_{i}"), "driver": "soapysdr"})
            except ImportError:
                pass

        return devices

    def _make_driver(self, device_index: int) -> _PythonHackrfWrapper | _SoapySdrWrapper:
        cfg = RadioConfig(
            center_frequency_hz=self.radio_config.center_frequency_hz,
            sample_rate_sps=self.radio_config.sample_rate_sps,
            tx_gain_db=self.radio_config.tx_gain_db,
            rx_gain_db=self.radio_config.rx_gain_db,
            frame_size=self.radio_config.frame_size,
            device_index=device_index,
        )
        try:
            return _PythonHackrfWrapper(cfg)
        except BackendUnavailableError:
            pass
        try:
            return _SoapySdrWrapper(cfg)
        except BackendUnavailableError as exc:
            raise BackendUnavailableError(
                f"Cannot open device {device_index} with any driver: {exc}"
            ) from exc

    def open(self) -> None:
        if self._opened:
            return

        self._tx_driver = self._make_driver(self.multi_config.tx_device_index)
        self._tx_driver.open()

        for rx_idx in self.multi_config.rx_device_indices:
            drv = self._make_driver(rx_idx)
            drv.open()
            self._rx_drivers.append(drv)

        self._opened = True

    def close(self) -> None:
        if self._tx_driver is not None:
            try:
                self._tx_driver.close()
            except Exception:
                pass
        for drv in self._rx_drivers:
            try:
                drv.close()
            except Exception:
                pass
        self._tx_driver = None
        self._rx_drivers.clear()
        self._opened = False

    def transmit(self, samples: np.ndarray) -> int:
        if not self._opened or self._tx_driver is None:
            raise BackendUnavailableError("MultiHackRF TX device not open")
        return self._tx_driver.transmit(samples)

    def receive(self, num_samples: int, time_step_s: float | None = None) -> IQFrame:
        if not self._opened or not self._rx_drivers:
            raise BackendUnavailableError("MultiHackRF has no RX devices")

        frames: list[IQFrame] = []
        for drv in self._rx_drivers:
            frames.append(drv.receive(num_samples, time_step_s))

        if len(frames) == 1:
            return frames[0]

        combined_samples = np.concatenate([f.samples for f in frames])
        ref = frames[0]
        return IQFrame(
            samples=combined_samples.astype(np.complex64, copy=False),
            timestamp_s=ref.timestamp_s,
            backend_name=f"multi_hackrf({len(frames)} rx)",
            center_frequency_hz=ref.center_frequency_hz,
            sample_rate_sps=ref.sample_rate_sps,
            metadata={f"rx_{i}_device_index": float(f.metadata.get("device_index", -1)) for i, f in enumerate(frames)},
        )
