"""Cluster-result repository port."""
from __future__ import annotations

from abc import ABC, abstractmethod

from domain.models import ClusterResult


class ClusterResultRepository(ABC):
    """Durable store for cluster results (e.g. a Postgres table)."""

    @abstractmethod
    def save(self, result: ClusterResult) -> None:
        ...
