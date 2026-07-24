"""Model-registry port."""
from __future__ import annotations

from abc import ABC, abstractmethod

from domain.models import ModelVersion


class ModelRegistry(ABC):
    """Lookup of registered model versions (e.g. the ``inference_model`` table)."""

    @abstractmethod
    def latest(self, model_type: str) -> ModelVersion:
        """Return the newest valid version for ``model_type``.

        Raises:
            LookupError: if no valid version exists for ``model_type``.
        """
        ...
