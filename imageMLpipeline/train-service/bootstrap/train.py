"""Training composition root.

Subscribes to the Kafka topic and runs the training use case on each 'train'
event. (Currently the use case just logs that an event was received.)

Run:  python -m bootstrap.train
"""
from __future__ import annotations

import logging

from adapters.inbound.train_consumer import TrainConsumer
from config.logging import configure_logging
from config.settings import Settings

logger = logging.getLogger("train-service")


def main() -> None:
    configure_logging()
    settings = Settings.from_env()

    consumer = TrainConsumer.initialize(settings)

    logger.info("Train service subscribed to topic '%s'", settings.kafka.topic)
    try:
        consumer.run()
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
