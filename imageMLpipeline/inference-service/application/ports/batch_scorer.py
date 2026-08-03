"""Batch anomaly-scoring port (stage-one).

Stage-one scoring is CPU-heavy — a ResNet50 forward pass plus a FAISS
nearest-neighbour search per image — so this port abstracts *where* that work
runs. The application hands it a batch of events and gets a verdict per event
back; whether the batch was scored in-process or spread over a Spark cluster is
the adapter's business.

Why the port is batch-shaped: distributing one image at a time is pure loss —
the scheduling round-trip and the per-executor model load dwarf the ~1s of
scoring. A batch amortises both.

Why only scalars come back: the adapter persists the result row and the heatmap
where the image was scored. Shipping a 1x3x224x224 preprocessed tensor back to
the driver just to render a PNG would move ~600 KB per image across the network
for no gain, so the render happens next to the data and only the verdict returns.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from domain.models import InferenceEvent


@dataclass(frozen=True)
class ScoredEvent:
    """What one image's scoring produced, as seen by the caller.

    ``status`` is the buffer-zone verdict (``normal`` / ``pending`` /
    ``anomaly``) and is ``None`` when scoring failed, in which case ``error``
    carries the reason. A failed image is reported, never silently dropped.
    """

    event: InferenceEvent
    status: str | None
    anomaly_score: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status is not None


class BatchAnomalyScorer(ABC):
    """Scores a batch of images, persisting each result where it was computed."""

    @abstractmethod
    def score(self, events: Sequence[InferenceEvent]) -> list[ScoredEvent]:
        """Score every event in ``events``, returning one result per event.

        The order of the returned list is not guaranteed to match the input;
        each :class:`ScoredEvent` carries its own event. Implementations must
        return a result for every input, including failures.
        """
        ...
