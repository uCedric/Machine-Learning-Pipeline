"""Inference composition root.

Subscribes to the Kafka topic, runs the model on each referenced image and
persists the result to Iceberg and a Postgres ``inference_results`` table.
Offsets are committed only after a result is saved (at-least-once).

Run:  python -m bootstrap.inference
"""
from __future__ import annotations

import logging

from adapters.inbound.inference_consumer import InferenceConsumer
from adapters.outbound.composite_repository import CompositeResultRepository
from adapters.outbound.iceberg_repository import IcebergResultRepository
from adapters.outbound.kafka_events import KafkaEventConsumer
from adapters.outbound.minio_storage import MinioObjectStorage
from adapters.outbound.patchcore.factory import build_patchcore
from adapters.outbound.patchcore.heatmap import MatplotlibHeatmapRenderer
from adapters.outbound.postgres_repository import PostgresResultRepository
from application.use_cases.run_inference import RunInferenceUseCase
from config.logging import configure_logging
from config.settings import Settings

logger = logging.getLogger("inference-service")


def main() -> None:
    configure_logging()
    settings = Settings.from_env()

    storage = MinioObjectStorage(settings.minio)
    model = build_patchcore(settings.model)
    renderer = MatplotlibHeatmapRenderer()
    repository = CompositeResultRepository(
        IcebergResultRepository(settings.iceberg),
        PostgresResultRepository(settings.iceberg.sql_uri),
    )
    use_case = RunInferenceUseCase(
        storage,
        model,
        renderer,
        repository,
        threshold=settings.model.threshold,
        heatmap_bucket=settings.minio.images_bucket,
    )
    consumer = KafkaEventConsumer(settings.kafka)

    logger.info("Inference service subscribed to topic '%s'", settings.kafka.topic)
    try:
        InferenceConsumer(consumer, use_case).run()
    finally:
        consumer.close()
        repository.close()


if __name__ == "__main__":
    main()
