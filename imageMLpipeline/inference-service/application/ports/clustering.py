"""Cluster-model port.

The application depends on this abstraction; the concrete DINOv2 + PCA/UMAP/
HDBSCAN adapter (or any future clustering model) implements it in
:mod:`adapters.outbound`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from domain.models import ClusterAssignment


class ClusterModel(ABC):
    """A clustering model that assigns a single image to a defect cluster."""

    @abstractmethod
    def assign(self, image: Any) -> ClusterAssignment:
        """Assign one image to a cluster.

        Returns a :class:`~domain.models.ClusterAssignment`; its ``cluster_id``
        is ``-1`` when the image matches no known cluster (noise).

        ``image`` may be a path or any file-like object the adapter can open.
        """
        ...
