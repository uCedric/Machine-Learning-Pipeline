"""Inference driving adapter: consume events and run the inference use case.

The concrete port implementations — object storage, the anomaly model, the
heatmap renderer, the result repository and the Kafka event consumer — are
wired here from :class:`Settings` via :meth:`InferenceConsumer.from_settings`.
The composition root only loads configuration and drives the loop.

Offsets are committed only after a result is saved (at-least-once delivery).
"""
from __future__ import annotations

import logging
from typing import Protocol

from adapters.outbound.kafka_events import KafkaEventConsumer
from adapters.outbound.minio_storage import MinioObjectStorage
from adapters.outbound.patchcore.factory import build_patchcore
from adapters.outbound.patchcore.heatmap import MatplotlibHeatmapRenderer
from adapters.outbound.postgres_repository import PostgresResultRepository
from application.ports.events import EventConsumer
from application.use_cases.run_inference import RunInferenceUseCase
from config.settings import Settings

logger = logging.getLogger("inference-service")


class _Closeable(Protocol):
    def close(self) -> None: ...


class InferenceConsumer:
    def __init__(
        self,
        consumer: EventConsumer,
        use_case: RunInferenceUseCase,
        *,
        closables: tuple[_Closeable, ...] = (),
    ) -> None:
        self._consumer = consumer
        self._use_case = use_case
        self._closables = closables

    @classmethod
    def initialize(cls, settings: Settings) -> "InferenceConsumer":
        """Build the port implementations from configuration and wire them up.

        Object storage, the anomaly model, the heatmap renderer and the result
        repository are constructed here and injected into the inference use
        case, alongside the Kafka event consumer that drives it. The consumer
        and repository are tracked as closables and released by :meth:`close`.
        """
        storage = MinioObjectStorage(settings.minio)
        model = build_patchcore(settings.model)
        renderer = MatplotlibHeatmapRenderer()
        repository = PostgresResultRepository(settings.postgres.sql_uri)
        use_case = RunInferenceUseCase(
            storage,
            model,
            renderer,
            repository,
            model_id=settings.model.model_id,
            buffer_zone=model.buffer_zone,
            heatmap_bucket=settings.minio.images_bucket,
        )
        consumer = KafkaEventConsumer(settings.kafka)
        return cls(consumer, use_case, closables=(consumer, repository))

    def run(self) -> None:
        for event in self._consumer.events():
            try:
                self._use_case.execute(event)
                self._consumer.commit()
            except Exception:
                logger.exception(
                    "Failed to process event for %s/%s",
                    event.bucket,
                    event.object_key,
                )

    def close(self) -> None:
        """Release the owned port implementations (event consumer, repository)."""
        for closable in self._closables:
            closable.close()
