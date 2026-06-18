"""Heatmap-renderer port.

The inference use case asks for an anomaly heatmap to be rendered and stored,
without knowing that the implementation uses matplotlib or where the bytes land.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from domain.models import BufferZone, PredictionStatus


class HeatmapRenderer(ABC):
    """Renders an anomaly heatmap for an image, returning the encoded bytes.

    Rendering is the technical concern (figure layout, image encoding). Deciding
    where the heatmap is stored belongs to the use case, so this port
    deliberately does not touch object storage.
    """

    @abstractmethod
    def render(
        self,
        original_img_np: Any,
        dist_score: Any,
        anomaly_score: float,
        buffer_zone: BufferZone,
        status: PredictionStatus,
        *,
        title: str | None = None,
    ) -> bytes:
        """Render the heatmap for one image and return it as encoded PNG bytes."""
        ...
