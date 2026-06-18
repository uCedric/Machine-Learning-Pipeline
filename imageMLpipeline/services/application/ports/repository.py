"""Result-repository port."""
from __future__ import annotations

from abc import ABC, abstractmethod

from domain.models import InferenceResult


class ResultRepository(ABC):
    """Durable store for inference results (e.g. a Postgres table)."""

    @abstractmethod
    def save(self, result: InferenceResult) -> None:
        ...
