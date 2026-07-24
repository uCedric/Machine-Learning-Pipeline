"""Model-registry port."""
from __future__ import annotations

from abc import ABC, abstractmethod

from domain.models import ModelVersion


class ModelRegistry(ABC):
    """Lookup and registration of model versions (the ``inference_model`` table)."""

    @abstractmethod
    def latest(self, model_type: str) -> ModelVersion:
        """Return the newest valid version for ``model_type`` (the retrain base).

        Raises:
            LookupError: if no valid version exists for ``model_type``.
        """
        ...

    @abstractmethod
    def next_version(self, base: ModelVersion) -> ModelVersion:
        """Derive the next ``patch + 1`` version off ``base`` with a fresh ``model_id``.

        The patch is computed as ``max(patch_version) + 1`` over all rows sharing
        ``base``'s ``(model_type, major, minor)`` — valid or not — so repeated
        retrains off the same base never collide on a version number.
        """
        ...

    @abstractmethod
    def register(self, version: ModelVersion, *, is_valid: bool) -> None:
        """Insert a new model version row (a retrain candidate)."""
        ...
