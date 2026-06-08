"""Inference composition root.

Subscribes to the Kafka topic, runs the model on each referenced image and
persists the result to Iceberg and a Postgres ``inference_results`` table.
Offsets are committed only after a result is saved (at-least-once).

The port implementations are wired inside
:meth:`adapters.inbound.inference_consumer.InferenceConsumer.from_settings`;
this root only loads configuration and runs the loop.

Run:  python -m bootstrap.inference
"""
from __future__ import annotations

import logging

from adapters.inbound.inference_consumer import InferenceConsumer
from config.logging import configure_logging
from config.settings import Settings

logger = logging.getLogger("inference-service")


def main() -> None:
    configure_logging()
    settings = Settings.from_env()

    consumer = InferenceConsumer.initialize(settings)

    logger.info("Inference service subscribed to topic '%s'", settings.kafka.topic)
    try:
        consumer.run()
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
