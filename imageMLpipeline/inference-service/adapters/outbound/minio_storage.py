"""MinIO adapter implementing the ObjectStorage port."""
from __future__ import annotations

import io
import logging

from minio import Minio

from application.ports.storage import ObjectStorage
from config.settings import MinioConfig
from domain.models import ImageObject

logger = logging.getLogger(__name__)


class MinioObjectStorage(ObjectStorage):
    def __init__(self, config: MinioConfig) -> None:
        self._client = Minio(
            config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure,
        )

    def _ensure_bucket(self, bucket: str) -> None:
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)
            logger.info("Created bucket '%s'", bucket)

    def put_object(self, bucket: str, key: str, data: bytes, content_type: str) -> ImageObject:
        self._ensure_bucket(bucket)
        size = len(data)
        self._client.put_object(
            bucket, key, io.BytesIO(data), length=size, content_type=content_type
        )
        return ImageObject(bucket=bucket, key=key, content_type=content_type, size_bytes=size)

    def get_object(self, bucket: str, key: str) -> bytes:
        response = None
        try:
            response = self._client.get_object(bucket, key)
            return response.read()
        finally:
            if response is not None:
                response.close()
                response.release_conn()
