"""Event-stream port (inbound)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from domain.models import TrainEvent


class EventConsumer(ABC):
    """Inbound event stream (e.g. a Kafka consumer) with explicit commit."""

    @abstractmethod
    def events(self) -> Iterator[TrainEvent]:
        ...

    @abstractmethod
    def commit(self) -> None:
        """Acknowledge that events yielded so far have been fully handled."""
        ...
