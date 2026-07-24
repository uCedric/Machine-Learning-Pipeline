"""Feature-embedder port.

The application depends on this abstraction; the concrete PatchCore embedder
(ONNX backbone + preprocessing transform) implements it in :mod:`adapters.outbound`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FeatureEmbedder(ABC):
    """Extracts the patch feature vectors for a single image."""

    @abstractmethod
    def embed(self, image: Any) -> Any:
        """Return the image's patch features as a ``(num_patches, dim)`` float32 array.

        ``image`` may be a path or any file-like object the adapter can open.
        """
        ...
