"""ELT composition root.

Loads a single image file into MinIO and emits an "inference" event to Kafka.

Run:  python -m bootstrap.elt
"""
from __future__ import annotations

import logging

from adapters.inbound.elt_poller import EltPoller
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
    publisher = KafkaEventPublisher(settings.kafka)
    use_case = IngestImage(storage, publisher, settings.minio.images_bucket)
    poller = EltPoller(
        use_case,
        settings.elt.input_file,
        poll_interval=settings.elt.poll_interval,
        run_once=settings.elt.run_once,
    )

    try:
        poller.run()
    finally:
        publisher.close()


if __name__ == "__main__":
    main()
