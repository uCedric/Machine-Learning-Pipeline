"""Clustering run-log port (stage-two).

Stage-two is gated on accumulation: a trigger only starts a clustering run once
enough new defect images have arrived since the last one finished. That needs a
memory of "last one finished", which is what this port provides — the watermark
to count from, and a place to record each run.

Deliberately *not* a counter. The number of pending defects is always derived
from ``inference_results``; this port only remembers how far the last completed
run reached. A stage-one replay that re-inserts result rows therefore cannot
drift the gate, because there is no second tally to drift.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from domain.models import ClusteringRun


class ClusteringRunLog(ABC):
    """Remembers when stage-two last clustered, and how far it covered."""

    @abstractmethod
    def last_watermark(self, mode: str | None = None) -> datetime | None:
        """``covered_through`` of the newest **completed** run, or ``None``.

        ``mode`` narrows it: ``None`` means any mode (the *assign* gate — how much
        has arrived since anything last ran), ``"fit"`` means only re-fits (the
        *fit* gate — how much has arrived since the clusters were last redrawn).

        The two must stay separate. If an assign advanced the fit watermark as
        well, the fit count would reset on every assign and never reach the re-fit
        threshold, so the clusters would never be redrawn.

        ``None`` means no such run has ever completed, so every defect image
        counts towards that gate. Failed runs are excluded on purpose: their work
        was never persisted, so their backlog must be retried.
        """
        ...

    @abstractmethod
    def record(self, run: ClusteringRun) -> None:
        """Append one finished run — completed or failed."""
        ...
