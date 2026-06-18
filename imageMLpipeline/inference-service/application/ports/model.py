"""Inference-model port.

The application depends on this abstraction; concrete detectors (the PatchCore
ONNX + FAISS adapter, or any future model) implement it in
:mod:`adapters.outbound`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple


class AnomalyModel(ABC):
    """An anomaly-detection model that scores a single image."""

    @abstractmethod
    def inference(self, image: Any, transform: Any = None) -> Tuple[Any, Any, float]:
        """Score one image.

        Returns ``(original_img_np, dist_score, anomaly_score)``:

        * ``original_img_np`` — the preprocessed image tensor (for visualisation).
        * ``dist_score``      — per-patch L2 distance map.
        * ``anomaly_score``   — the image-level score (max patch distance).

        ``image`` may be a path or any file-like object the adapter can open.
        """
        ...
