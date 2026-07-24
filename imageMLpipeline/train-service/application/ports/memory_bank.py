"""Memory-bank repository port.

Encapsulates reading/writing a model version's memory bank and copying the
version's unchanged assets (backbone, buffer zone) so a retrain produces a
self-contained version directory in object storage.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from domain.models import ModelVersion


class MemoryBankRepository(ABC):
    @abstractmethod
    def load(self, version: ModelVersion) -> Any:
        """Return ``version``'s memory bank as a ``(n, dim)`` float32 array."""
        ...

    @abstractmethod
    def save(self, version: ModelVersion, bank: Any) -> None:
        """Persist ``bank`` as ``version``'s memory bank."""
        ...

    @abstractmethod
    def save_buffer_zone(
        self, version: ModelVersion, lower: float, upper: float
    ) -> None:
        """Persist the recomputed anomaly-score buffer zone for ``version``."""
        ...

    @abstractmethod
    def copy_unchanged_assets(self, src: ModelVersion, dst: ModelVersion) -> None:
        """Copy the assets a retrain does not change (the backbone) from ``src``'s
        version directory to ``dst``'s, making it self-contained."""
        ...

    @abstractmethod
    def copy_buffer_zone(self, src: ModelVersion, dst: ModelVersion) -> None:
        """Copy ``src``'s buffer zone to ``dst`` verbatim — the fallback when a
        retrain has no validation images to recompute the bounds from."""
        ...
