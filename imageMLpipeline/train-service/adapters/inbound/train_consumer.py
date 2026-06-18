"""Training driving adapter: consume 'train' events and run the training use case.

The concrete port implementation — the Kafka event consumer — is wired here from
:class:`Settings` via :meth:`TrainConsumer.initialize`. The composition root only
loads configuration and drives the loop.

Offsets are committed only after the use case handles the event (at-least-once).
"""
from __future__ import annotations

import logging
from typing import Protocol

from adapters.outbound.kafka_events import KafkaEventConsumer
from application.ports.events import EventConsumer
from application.use_cases.run_training import RunTraining
from config.settings import Settings

logger = logging.getLogger("train-service")


class _Closeable(Protocol):
    def close(self) -> None: ...


class TrainConsumer:
    def __init__(
        self,
        consumer: EventConsumer,
        use_case: RunTraining,
        *,
        closables: tuple[_Closeable, ...] = (),
    ) -> None:
        self._consumer = consumer
        self._use_case = use_case
        self._closables = closables

    @classmethod
    def initialize(cls, settings: Settings) -> "TrainConsumer":
        """Build the port implementation from configuration and wire it up."""
        consumer = KafkaEventConsumer(settings.kafka)
        use_case = RunTraining()
        return cls(consumer, use_case, closables=(consumer,))

    def run(self) -> None:
        for event in self._consumer.events():
            try:
                self._use_case.execute(event)
                self._consumer.commit()
            except Exception:
                logger.exception("Failed to process train event")

    def close(self) -> None:
        """Release the owned port implementations (event consumer)."""
        for closable in self._closables:
            closable.close()
