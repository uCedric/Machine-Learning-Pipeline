"""Training driving adapter: consume 'train' events and run the training use case.

The concrete port implementations — object storage, the model registry, the
feature embedder and the memory-bank repository — are wired here from
:class:`Settings` via :meth:`TrainConsumer.initialize`, alongside the Kafka event
consumer that drives the loop. The composition root only loads configuration and
runs the loop.

Offsets are committed only after the use case handles the event (at-least-once).
"""
from __future__ import annotations

import logging
from typing import Protocol

from adapters.outbound.kafka_events import KafkaEventConsumer
from adapters.outbound.minio_storage import MinioObjectStorage
from adapters.outbound.patchcore.embedder import PatchCoreEmbedder
from adapters.outbound.patchcore.memory_bank_store import PatchCoreMemoryBankStore
from adapters.outbound.patchcore.resources import build_transform, load_onnx_session
from adapters.outbound.postgres_model_registry import PostgresModelRegistry
from adapters.outbound.postgres_image_repository import PostgresImageRepository
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
        """Build the port implementations from configuration and wire them up.

        The ONNX backbone is downloaded once at start-up from the newest valid
        model version (it is identical across retrains, since a retrain only
        extends the memory bank). The registry stays open because the use case
        reads and registers versions on every event.
        """
        storage = MinioObjectStorage(settings.minio)
        registry = PostgresModelRegistry(settings.postgres.sql_uri)

        # Locate and load the backbone of the newest valid version.
        base = registry.latest(settings.model.model_key)
        session = load_onnx_session(
            storage, base.bucket, base.model_type, base.version, settings.model.cache_dir
        )
        embedder = PatchCoreEmbedder(session, build_transform(settings.model.image_size))
        memory_bank = PatchCoreMemoryBankStore(storage)
        repository = PostgresImageRepository(settings.postgres.sql_uri)

        use_case = RunTraining(
            registry,
            embedder,
            memory_bank,
            repository,
            storage_get=storage.get_object,
            validation_set_size=settings.model.validation_set_size,
        )
        consumer = KafkaEventConsumer(settings.kafka)
        return cls(consumer, use_case, closables=(consumer, registry, repository))

    def run(self) -> None:
        for event in self._consumer.events():
            try:
                self._use_case.execute(event)
                self._consumer.commit()
            except Exception:
                logger.exception("Failed to process train event")

    def close(self) -> None:
        """Release the owned port implementations (event consumer, registry)."""
        for closable in self._closables:
            closable.close()
