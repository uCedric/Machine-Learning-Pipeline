"""Kafka adapter implementing the EventConsumer port."""
from __future__ import annotations

import json
import logging
from typing import Iterator

from kafka import KafkaConsumer

from application.ports.events import EventConsumer
from config.settings import KafkaConfig
from domain.models import TrainEvent

logger = logging.getLogger(__name__)


class KafkaEventConsumer(EventConsumer):
    def __init__(self, config: KafkaConfig) -> None:
        self._event_type = config.event_type
        self._consumer = KafkaConsumer(
            config.topic,
            bootstrap_servers=config.bootstrap_servers,
            group_id=config.group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )

    def events(self) -> Iterator[TrainEvent]:
        for message in self._consumer:
            payload = message.value
            if payload.get("event") != self._event_type:
                logger.warning("Skipping non-train message at offset %s", message.offset)
                self.commit()
                continue
            yield TrainEvent(event=payload["event"], payload=payload)

    def commit(self) -> None:
        self._consumer.commit()

    def close(self) -> None:
        self._consumer.close()
