"""Domain entities and value objects.

These are plain, immutable data structures with no dependency on any framework
or external SDK. They are the vocabulary the rest of the application speaks and
sit at the centre of the hexagon — nothing here imports an adapter, a port or
configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

INFERENCE_EVENT = "inference"


def _new_id() -> str:
    return uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ImageObject:
    """An image that lives in object storage."""

    bucket: str
    key: str
    content_type: str
    size_bytes: int


@dataclass(frozen=True)
class InferenceEvent:
    """Signals that a stored image is ready to be inferred.

    Emitted by the ELT once an image lands in object storage and consumed by the
    inference service.
    """

    bucket: str
    object_key: str
    content_type: str
    size_bytes: int
    event: str = INFERENCE_EVENT
    event_id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class InferenceResult:
    """The outcome of running the model against a single image."""

    event_id: str
    bucket: str
    object_key: str
    anomaly_score: float
    is_anomaly: bool
    model_name: str
    heatmap_key: str
    inferred_at: datetime = field(default_factory=_utcnow)
