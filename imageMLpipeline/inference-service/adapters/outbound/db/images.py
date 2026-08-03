"""``image`` table: a record of every image the ELT landed in object storage.

Each :class:`~domain.models.ImageObject` is keyed by the image id (the object
key's uuid stem). The row must exist before the inference service writes its
result, since ``inference_results.image_id`` references ``image.image_id``.

One adapter writes it: :class:`PostgresImageRepository`, the ELT's record of
every ingested image.

The table is provisioned by the Flyway migrations in ``migrations/`` (see
``V1__init.sql``), so it must exist before the adapters are used.
"""
from __future__ import annotations

import logging
from pathlib import PurePosixPath

from adapters.outbound.db.repository import PostgresRepository
from application.ports.image_repository import ImageRepository
from domain.models import ImageObject

logger = logging.getLogger(__name__)

_INSERT_SQL = """
INSERT INTO image
    (image_id, object_key, bucket, content_type, size_bytes)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (image_id) DO NOTHING
"""

class PostgresImageRepository(PostgresRepository, ImageRepository):
    """Stores :class:`~domain.models.ImageObject` rows in Postgres."""

    def save(self, image: ImageObject) -> None:
        # The object key is a uuid-based file name; the image id is that uuid
        # (the key without its extension), recording one row per image.
        image_id = PurePosixPath(image.key).stem
        self._execute(
            _INSERT_SQL,
            (
                image_id,
                image.key,
                image.bucket,
                image.content_type,
                image.size_bytes,
            ),
        )
