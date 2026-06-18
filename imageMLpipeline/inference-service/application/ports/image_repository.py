"""Image-repository port."""
from __future__ import annotations

from abc import ABC, abstractmethod

from domain.models import ImageObject


class ImageRepository(ABC):
    """Durable record of every image landed in object storage (e.g. Postgres)."""

    @abstractmethod
    def save(self, image: ImageObject) -> None:
        ...
