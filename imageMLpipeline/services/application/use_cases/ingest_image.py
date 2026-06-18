"""ELT use case: load an image into object storage, then emit an inference event.

Pure application logic — it depends only on the :mod:`application.ports`
abstractions and the :mod:`domain` model, never on a concrete SDK.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from application.ports.events import EventPublisher
from application.ports.image_repository import ImageRepository
from application.ports.storage import ObjectStorage
from domain.models import ImageObject, InferenceEvent

logger = logging.getLogger(__name__)

_DEFAULT_DEAD_LETTER_DIR = "failed_events"


class IngestImage:
    def __init__(
        self,
        storage: ObjectStorage,
        image_repository: ImageRepository,
        publisher: EventPublisher,
        bucket: str,
        topic: str,
        event_type: str,
        dead_letter_dir: str = _DEFAULT_DEAD_LETTER_DIR,
    ) -> None:
        self._storage = storage
        self._image_repository = image_repository
        self._publisher = publisher
        self._bucket = bucket
        self._topic = topic
        self._event_type = event_type
        self._dead_letter_dir = Path(dead_letter_dir)

    def execute(self, key: str, data: bytes, content_type: str) -> InferenceEvent:
        try:
            stored: ImageObject = self._storage.put_object(self._bucket, key, data, content_type)
        except Exception:
            # Upload failed: do not publish an inference event for an object that
            # never landed. Log with context and re-raise for the caller to handle.
            logger.exception(
                "Failed to store image %s/%s (%d bytes)", self._bucket, key, len(data)
            )
            raise
        logger.info(
            "Loaded image %s/%s (%d bytes)", stored.bucket, stored.key, stored.size_bytes
        )

        try:
            # Record the image before announcing it: the inference result row
            # references this image (FK on image_id), so the row must exist
            # before the inference service consumes the event.
            self._image_repository.save(stored)
        except Exception:
            # Stored in MinIO but not recorded: publishing now would let the
            # inference service produce a result that violates the FK. Do not
            # publish; log with context and re-raise for the caller to reconcile.
            logger.exception(
                "Stored image %s/%s but failed to record it in the image table",
                stored.bucket,
                stored.key,
            )
            raise

        event = InferenceEvent(
            bucket=stored.bucket,
            object_key=stored.key,
            content_type=stored.content_type,
            size_bytes=stored.size_bytes,
            event=self._event_type,
        )
        try:
            self._publisher.publish(self._topic, event)
        except Exception:
            # Image is stored but the event was not emitted: the object now exists
            # in MinIO with no downstream notification. Log with context and
            # re-raise so the caller can reconcile / retry.
            logger.exception(
                "Stored image %s/%s but failed to publish '%s' event",
                event.bucket,
                event.object_key,
                event.event,
            )
            self._write_dead_letter(event)
            raise
        logger.info(
            "Published '%s' event for %s/%s",
            event.event,
            event.bucket,
            event.object_key,
        )
        return event

    def _write_dead_letter(self, event: InferenceEvent) -> None:
        """Persist an unpublished event to a local .json file for later reconciliation.

        Uses the same payload shape as the Kafka publisher so the file can be
        reloaded and re-published as-is.
        """
        payload = {
            "event": event.event,
            "bucket": event.bucket,
            "object_key": event.object_key,
            "content_type": event.content_type,
            "size_bytes": event.size_bytes,
            "created_at": event.created_at.isoformat(),
        }
        try:
            self._dead_letter_dir.mkdir(parents=True, exist_ok=True)
            path = self._dead_letter_dir / f"{Path(event.object_key).stem}.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.warning("Wrote unpublished event %s to dead-letter file %s", event.object_key, path)
        except OSError:
            # A dead-letter write failure must not mask the original publish error.
            logger.exception("Failed to write dead-letter file for event %s", event.object_key)
