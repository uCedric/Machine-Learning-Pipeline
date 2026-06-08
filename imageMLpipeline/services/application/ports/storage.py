"""Object-storage port."""
from __future__ import annotations

from abc import ABC, abstractmethod

from domain.models import ImageObject


class ObjectStorage(ABC):
    """Binary blob storage (e.g. S3 / MinIO)."""

    @abstractmethod
    def put_object(self, bucket: str, key: str, data: bytes, content_type: str) -> ImageObject:
        ...

    @abstractmethod
    def get_object(self, bucket: str, key: str) -> bytes:
        ...
