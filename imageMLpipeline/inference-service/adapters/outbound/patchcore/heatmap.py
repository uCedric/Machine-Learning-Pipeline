"""Heatmap renderer (stage 3) — a concrete
:class:`~application.ports.heatmap.HeatmapRenderer`.

Turns PatchCore's per-patch distance map into a two-panel comparison figure and
returns it as encoded PNG bytes. Persisting those bytes is the use case's job.

Uses matplotlib's object-oriented ``Figure`` API (not ``pyplot``) so it carries
no global state and is safe to call concurrently from multiple threads.
"""
from __future__ import annotations

import io
import logging

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from application.ports.heatmap import HeatmapRenderer
from domain.models import BufferZone, PredictionStatus

logger = logging.getLogger(__name__)


class MatplotlibHeatmapRenderer(HeatmapRenderer):
    def __init__(
        self,
        *,
        grid_size: int = 28,
        image_size: int = 224,
    ) -> None:
        self._grid_size = grid_size
        self._image_size = image_size

    def render(
        self,
        original_img_np: np.ndarray,
        dist_score: np.ndarray,
        anomaly_score: float,
        buffer_zone: BufferZone,
        status: PredictionStatus,
        *,
        title: str | None = None,
    ) -> bytes:
        """Build the segmentation heatmap and return it as encoded PNG bytes.

        ``dist_score`` is the per-patch L2 distance map
        (``grid_size * grid_size`` values); ``original_img_np`` is the
        preprocessed image tensor produced by the inference stage.
        """
        # Reshape to the backbone's spatial grid and upscale to input resolution.
        segm_map = torch.from_numpy(dist_score).view(1, 1, self._grid_size, self._grid_size)
        segm_map = (
            torch.nn.functional.interpolate(
                segm_map, size=(self._image_size, self._image_size), mode="bilinear"
            )
            .squeeze()
            .numpy()
        )

        # Render the two-panel comparison figure (to a buffer, not a screen).
        # Object-oriented Figure + explicit Agg canvas: no pyplot global state,
        # so concurrent renders from multiple threads don't corrupt each other.
        fig = Figure(figsize=(12, 5))
        FigureCanvasAgg(fig)  # attaches itself as fig.canvas

        ax1 = fig.add_subplot(1, 2, 1)
        # Left panel shows the preprocessed (normalised) tensor, as in the original.
        ax1.imshow(original_img_np.squeeze().transpose(1, 2, 0))
        ax1.set_title(title or "Test Image")

        # Anchor the colour scale across the buffer zone so heatmaps stay
        # comparable image-to-image (rather than self-normalising per image).
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(
            segm_map, cmap="jet", vmin=buffer_zone.lower * 0.8, vmax=buffer_zone.upper * 1.2
        )
        ax2.set_title(f"Score: {anomaly_score:.2f} | {status.upper()}")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        # No plt.close() needed: the Figure is not registered in any global
        # manager, so it is reclaimed by normal GC when this method returns.

        png = buf.getvalue()
        logger.debug("Rendered heatmap (%d bytes)", len(png))
        return png
