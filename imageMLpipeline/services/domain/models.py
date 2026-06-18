"""Domain entities and value objects.

These are plain, immutable data structures with no dependency on any framework
or external SDK. They are the vocabulary the rest of the application speaks and
sit at the centre of the hexagon — nothing here imports an adapter, a port or
configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PredictionStatus(StrEnum):
    """The verdict for an inferred image.

    ``StrEnum`` so the value *is* its lowercase string (``"normal"`` etc.),
    matching the ``prediction_status`` Postgres enum and serialising cleanly.
    """

    NORMAL = "normal"
    PENDING = "pending"
    ANOMALY = "anomaly"


@dataclass(frozen=True)
class BufferZone:
    """A two-bound decision band over the anomaly score.

    Replaces a single threshold with a hysteresis-style buffer: scores below
    ``lower`` are clearly normal, scores above ``upper`` are clearly anomalous,
    and scores inside ``[lower, upper]`` are uncertain (``PENDING``) and worth a
    human look. The bounds are inclusive of the pending band.
    """

    lower: float
    upper: float

    def classify(self, anomaly_score: float) -> PredictionStatus:
        if anomaly_score < self.lower:
            return PredictionStatus.NORMAL
        if anomaly_score > self.upper:
            return PredictionStatus.ANOMALY
        return PredictionStatus.PENDING


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
    inference service. The image is identified by its ``object_key`` — a
    uuid-based file name — so no separate event id is carried.
    """

    bucket: str
    object_key: str
    content_type: str
    size_bytes: int
    event: str
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class InferenceResult:
    """The outcome of running the model against a single image.

    Identified by its own ``event_id`` (one row per inference execution). The
    source image is ``object_key`` (a uuid-based file name) and the model is
    referenced by ``model_id``.
    """

    bucket: str
    object_key: str
    anomaly_score: float
    status: PredictionStatus
    model_id: str
    heatmap_key: str
    event_id: str = field(default_factory=lambda: str(uuid4()))
    inferred_at: datetime = field(default_factory=_utcnow)
