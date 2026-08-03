"""Model-registry port."""
from __future__ import annotations

from abc import ABC, abstractmethod

from domain.models import ModelVersion


class ModelRegistry(ABC):
    """Registered model versions (e.g. the ``inference_model`` table)."""

    @abstractmethod
    def latest(self, model_type: str) -> ModelVersion:
        """Return the newest valid version for ``model_type``.

        Raises:
            LookupError: if no valid version exists for ``model_type``.
        """
        ...

    @abstractmethod
    def next_minor(self, model_type: str) -> ModelVersion:
        """Compute the next minor version for ``model_type`` **without storing it**.

        Splitting this from :meth:`register` is deliberate: the caller needs the
        version — and therefore the object-storage prefix — *before* it can upload
        the artifacts, and the registry row must not appear until those artifacts
        are actually there. A row pointing at a prefix that does not exist would
        make every later load fail.

        Raises:
            LookupError: if no valid version exists to bump from.
        """
        ...

    @abstractmethod
    def register(self, version: ModelVersion) -> None:
        """Store ``version`` as a valid registered version.

        Valid immediately, unlike the train service's candidate versions: a
        stage-two fit is not a new *backbone* awaiting evaluation, it is the
        deployed backbone re-fitted on newer data, so nothing gates it.
        """
        ...
