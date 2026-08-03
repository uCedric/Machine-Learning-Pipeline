"""Kafka adapters implementing the EventPublisher and EventConsumer ports."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Iterator

from kafka import KafkaConsumer, KafkaProducer

from application.ports.events import EventConsumer, EventPublisher
from config.settings import KafkaConfig
from domain.models import InferenceEvent

logger = logging.getLogger(__name__)


def _to_payload(event: InferenceEvent) -> Dict[str, Any]:
    return {
        "event": event.event,
        "type": str(event.type),
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
        event=payload["event"],
        # Default to stage-one so an event without the discriminator still routes.
        type=payload.get("type", "stage-one"),
        created_at=datetime.fromisoformat(payload["created_at"]),
    )


class KafkaEventPublisher(EventPublisher):
    def __init__(self, config: KafkaConfig) -> None:
        self._producer = KafkaProducer(
            bootstrap_servers=config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
        )

    def publish(self, topic: str, event: InferenceEvent) -> None:
        future = self._producer.send(topic, key=event.event, value=_to_payload(event))
        future.get(timeout=10)

    def close(self) -> None:
        self._producer.flush()
        self._producer.close()


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

    def events(self) -> Iterator[InferenceEvent]:
        for message in self._consumer:
            payload = message.value
            if payload.get("event") != self._event_type:
                logger.warning(
                    "Skipping message with event=%r (expected %r) at offset %s",
                    payload.get("event"),
                    self._event_type,
                    message.offset,
                )
                self.commit()
                continue
            yield _from_payload(payload)

    def poll(self, max_records: int, timeout_ms: int) -> list[InferenceEvent]:
        """Accumulate up to ``max_records`` events for one dispatch round.

        ``KafkaConsumer.poll`` returns a ``{TopicPartition: [records]}`` map, so
        the partition batches are flattened back into arrival order. Events of a
        foreign ``event`` type are skipped here exactly as in :meth:`events`.
        """
        batches = self._consumer.poll(timeout_ms=timeout_ms, max_records=max_records)
        events: list[InferenceEvent] = []
        for records in batches.values():
            for message in records:
                payload = message.value
                if payload.get("event") != self._event_type:
                    logger.warning(
                        "Skipping message with event=%r (expected %r) at offset %s",
                        payload.get("event"),
                        self._event_type,
                        message.offset,
                    )
                    continue
                events.append(_from_payload(payload))
        return events

    def commit(self) -> None:
        self._consumer.commit()

    def close(self) -> None:
        self._consumer.close()
