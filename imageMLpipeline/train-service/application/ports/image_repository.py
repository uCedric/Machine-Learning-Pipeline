"""Validation-set port.

A retrain recomputes the anomaly-score buffer zone from a *rolling validation
set* of recently-seen known-good images. This port abstracts where that set
comes from; the concrete adapter reads it from the inference-results store.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ImageRepository(ABC):
    """Source of recent known-good images used to recalibrate the buffer zone."""

    @abstractmethod
    def recent_good(self, model_type: str, limit: int) -> list[tuple[str, str]]:
        """Return ``(bucket, object_key)`` for the most recent known-good images
        of ``model_type`` (newest first), at most ``limit`` of them.

        "Known-good" is an image a prior inference classified as ``normal``.
        Returns an empty list when none are available.
        """
        ...
