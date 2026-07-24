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
from adapters.outbound.factory import ModelFactory
from adapters.outbound.patchcore.heatmap import MatplotlibHeatmapRenderer
from adapters.outbound.postgres_cluster_repository import PostgresClusterResultRepository
from adapters.outbound.postgres_model_registry import PostgresModelRegistry
from adapters.outbound.postgres_repository import PostgresResultRepository
from application.ports.events import EventConsumer
from application.use_cases.run_inference import RunInferenceUseCase
from config.settings import ModelConfig, Settings

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

        # initialize registered model
        storage = MinioObjectStorage(settings.minio)
        # Resolve the newest registered model version, then load it. The registry
        # is only needed at start-up, so it is closed once the model is built.
        registry = PostgresModelRegistry(settings.postgres.sql_uri)
        try:
            factory = ModelFactory(registry, storage)
            model_config = ModelConfig(
                name=settings.model.name,
                model_key=settings.model.model_key,
                image_size=settings.model.image_size,
                cache_dir=settings.model.cache_dir,
            )
            model = factory.build(model_config)

            # Second model: DINOv2 defect clustering. Its inference_model row
            # is the activation gate — until an operator registers a valid
            # version (assets uploaded to the models bucket), LookupError
            # disables clustering and inference runs exactly as before. Any
            # other failure (broken or missing assets) stays loud, like the
            # anomaly model's.
            cluster_model = None
            try:
                cluster_model = factory.build(settings.cluster_model)
            except LookupError:
                logger.info(
                    "No valid '%s' model registered; defect clustering disabled",
                    settings.cluster_model.model_key,
                )
        finally:
            registry.close()

        # prepare dependcies for the inference use case and inject them
        renderer = MatplotlibHeatmapRenderer()
        repository = PostgresResultRepository(settings.postgres.sql_uri)
        # The cluster repository (and its connection pool) only exists when the
        # clustering stage is active.
        cluster_repository = (
            PostgresClusterResultRepository(settings.postgres.sql_uri)
            if cluster_model is not None
            else None
        )
        use_case = RunInferenceUseCase(
            storage,
            model,
            renderer,
            repository,
            model_id=model.model_id,
            buffer_zone=model.buffer_zone,
            heatmap_bucket=settings.minio.images_bucket,
            cluster_model=cluster_model,
            cluster_repository=cluster_repository,
        )
        consumer = KafkaEventConsumer(settings.kafka)
        closables: tuple[_Closeable, ...] = (consumer, repository)
        if cluster_repository is not None:
            closables += (cluster_repository,)
        return cls(consumer, use_case, closables=closables)

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
