"""Event-stream ports (inbound and outbound)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from domain.models import InferenceEvent


class EventPublisher(ABC):
    """Outbound event stream (e.g. a Kafka producer)."""

    @abstractmethod
    def publish(self, topic: str, event: InferenceEvent) -> None:
        ...


class EventConsumer(ABC):
    """Inbound event stream (e.g. a Kafka consumer) with explicit commit."""

    @abstractmethod
    def events(self) -> Iterator[InferenceEvent]:
        ...

    @abstractmethod
    def commit(self) -> None:
        """Acknowledge that events yielded so far have been fully handled."""
        ...
