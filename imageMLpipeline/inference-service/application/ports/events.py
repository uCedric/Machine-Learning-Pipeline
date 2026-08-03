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
    def poll(self, max_records: int, timeout_ms: int) -> list[InferenceEvent]:
        """Return up to ``max_records`` events, waiting at most ``timeout_ms``.

        The batch-shaped counterpart to :meth:`events`. Stage-one scoring is
        distributed, and distributing one image at a time is pure loss, so the
        consumer accumulates a micro-batch before dispatching it. Returns an
        empty list when the timeout expires with nothing available, which is the
        idle case and not an error.
        """
        ...

    @abstractmethod
    def commit(self) -> None:
        """Acknowledge that events yielded so far have been fully handled."""
        ...
