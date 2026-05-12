"""Radio backend abstractions used by the session manager."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class BackendUnavailableError(RuntimeError):
    """Raised when a hardware backend cannot be opened."""


@dataclass(slots=True)
class IQFrame:
    """Container for one captured complex sample frame."""

    samples: np.ndarray
    timestamp_s: float
    backend_name: str
    center_frequency_hz: float
    sample_rate_sps: float
    metadata: dict[str, float] = field(default_factory=dict)


class RadioBackend(ABC):
    """Common interface for simulation and physical radios."""

    name: str = "radio"

    @abstractmethod
    def open(self) -> None:
        """Open the hardware or simulation backend."""

    @abstractmethod
    def close(self) -> None:
        """Close the backend and release any resources."""

    @abstractmethod
    def transmit(self, samples: np.ndarray) -> int:
        """Transmit a complex baseband chunk and return the accepted sample count."""

    @abstractmethod
    def receive(self, num_samples: int, time_step_s: float | None = None) -> IQFrame:
        """Receive a complex frame from the backend."""

