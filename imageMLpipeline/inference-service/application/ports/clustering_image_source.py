"""Clustering image-source port (stage-two).

Stage-two clusters a *set* of images, so — unlike the per-image inference flow —
it needs to enumerate two cohorts from the recorded predictions: the **defect
set** to cluster and the **known-good baseline** the descriptors are measured
against. Both are read from the ``inference_results`` stage-one wrote.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime


class ClusteringImageSource(ABC):
    """Enumerates the image cohorts a stage-two batch clustering run needs."""

    @abstractmethod
    def anomalous(self, limit: int) -> list[tuple[str, str]]:
        """The most recent distinct defect images, as ``(bucket, object_key)``.

        Defect = a stage-one verdict of ``anomaly`` — a *confirmed* defect. The
        uncertain ``pending`` band is excluded: those images are for human review
        (and retraining) rather than for an unsupervised grouping that is meant
        to name defect types. Newest first, at most ``limit``.
        """
        ...

    @abstractmethod
    def recent_good(self, model_type: str, limit: int) -> list[tuple[str, str]]:
        """The most recent distinct known-good (``normal``) images for ``model_type``.

        Used to build the normal texture baseline. Newest first, at most ``limit``.
        """
        ...

    @abstractmethod
    def anomalous_since(
        self, since: datetime | None, limit: int
    ) -> list[tuple[str, str]]:
        """The distinct defect images that arrived after ``since``, newest first.

        What an *assign* run works on: only the newly arrived defects, because the
        older ones already carry a ``cluster_id`` from the fit (or from an earlier
        assign against the same frozen state) and re-labelling them would just
        rewrite identical rows.
        """
        ...

    @abstractmethod
    def backlog_since(self, since: datetime | None) -> tuple[int, datetime | None]:
        """How many ``anomaly`` images arrived after ``since``, and the newest one's time.

        The accumulation gate: stage-two runs only once enough *new* defects have
        landed since the last completed run. ``since`` of ``None`` means "count
        everything" (no run has completed yet).

        Returns ``(count, newest_inferred_at)``. The timestamp is the watermark
        the next run records, so results written while a run is in flight are
        counted towards the *following* run rather than being skipped. It is
        ``None`` exactly when the count is zero.

        Counts rows, not distinct images: an image re-scored twice counts twice.
        That is deliberate — the gate measures "how much new evidence has arrived",
        and the clustering step dedupes by ``object_key`` on its own.
        """
        ...
