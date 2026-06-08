"""Kafka adapters implementing the EventPublisher and EventConsumer ports."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Iterator

from kafka import KafkaConsumer, KafkaProducer

from application.ports.events import EventConsumer, EventPublisher
from config.settings import KafkaConfig
from domain.models import INFERENCE_EVENT, InferenceEvent

logger = logging.getLogger(__name__)


def _to_payload(event: InferenceEvent) -> Dict[str, Any]:
    return {
        "event": event.event,
        "bucket": event.bucket,
        "object_key": event.object_key,
        "content_type": event.content_type,
        "size_bytes": event.size_bytes,
        "created_at": event.created_at.isoformat(),
    }


def _from_payload(payload: Dict[str, Any]) -> InferenceEvent:
    return InferenceEvent(
        bucket=payload["bucket"],
        object_key=payload["object_key"],
        content_type=payload.get("content_type", "application/octet-stream"),
        size_bytes=int(payload.get("size_bytes", 0)),
        event=payload.get("event", INFERENCE_EVENT),
        created_at=datetime.fromisoformat(payload["created_at"]),
    )


class KafkaEventPublisher(EventPublisher):
    def __init__(self, config: KafkaConfig) -> None:
        self._topic = config.topic
        self._producer = KafkaProducer(
            bootstrap_servers=config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
        )

    def publish(self, event: InferenceEvent) -> None:
        future = self._producer.send(self._topic, key=event.event, value=_to_payload(event))
        future.get(timeout=10)

    def close(self) -> None:
        self._producer.flush()
        self._producer.close()


class KafkaEventConsumer(EventConsumer):
    def __init__(self, config: KafkaConfig) -> None:
        self._consumer = KafkaConsumer(
            config.topic,
            bootstrap_servers=config.bootstrap_servers,
            group_id=config.group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )

    def events(self) -> Iterator[InferenceEvent]:
        for message in self._consumer:
            payload = message.value
            if payload.get("event") != INFERENCE_EVENT:
                logger.warning("Skipping non-inference message at offset %s", message.offset)
                self.commit()
                continue
            yield _from_payload(payload)

    def commit(self) -> None:
        self._consumer.commit()

    def close(self) -> None:
        self._consumer.close()
