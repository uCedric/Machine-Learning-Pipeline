"""ELT composition root.

Loads a single image file into MinIO and emits a "stage-one-inference" event to Kafka.

Run:  python -m bootstrap.elt
"""
from __future__ import annotations

import logging

from adapters.inbound.elt_poller import EltPoller
from adapters.outbound.db import Database, PostgresImageRepository
from adapters.outbound.kafka_events import KafkaEventPublisher
from adapters.outbound.minio_storage import MinioObjectStorage
from application.use_cases.ingest_image import IngestImage
from config.logging import configure_logging
from config.settings import Settings

logger = logging.getLogger("elt")


def main() -> None:
    configure_logging()
    settings = Settings.from_env()

    storage = MinioObjectStorage(settings.minio)
    # One pool for the process; the repository borrows from it.
    database = Database(settings.postgres.sql_uri)
    image_repository = PostgresImageRepository(database)
    publisher = KafkaEventPublisher(settings.kafka)
    use_case = IngestImage(
        storage,
        image_repository,
        publisher,
        settings.minio.images_bucket,
        topic=settings.kafka.topic,
        event_type=settings.kafka.event_type,
    )
    poller = EltPoller(
        use_case,
        settings.elt.input_folder,
        poll_interval=settings.elt.poll_interval,
        run_once=settings.elt.run_once,
    )

    try:
        poller.run()
    finally:
        publisher.close()
        database.close()


if __name__ == "__main__":
    main()
