"""Postgres adapter: record every ingested image in a relational table.

Each :class:`~domain.models.ImageObject` the ELT lands in MinIO is written to the
``image`` table, keyed by the image id (the object key's uuid stem). The row must
exist before the inference service writes its result, since
``inference_results.image_id`` references ``image.image_id``.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import PurePosixPath
from typing import Iterator

from psycopg2.extensions import connection as Connection
from psycopg2.pool import ThreadedConnectionPool

from application.ports.image_repository import ImageRepository
from domain.models import ImageObject

logger = logging.getLogger(__name__)


class PostgresImageRepository(ImageRepository):
    """Stores :class:`~domain.models.ImageObject` rows in Postgres.

    The ``image`` table is provisioned by the Flyway migrations in ``migrations/``
    (see ``V1__init.sql``), so it must exist before the adapter is used.
    """

    def __init__(self, sql_uri: str, minconn: int = 1, maxconn: int = 10) -> None:
        # ``sql_uri`` is a SQLAlchemy URL (``postgresql+psycopg2://``); psycopg2
        # wants a plain libpq DSN, so drop the driver suffix.
        dsn = sql_uri.replace("postgresql+psycopg2://", "postgresql://")
        self._pool = ThreadedConnectionPool(minconn, maxconn, dsn)
        logger.info("Postgres image connection pool ready")

    @contextmanager
    def _connection(self) -> Iterator[Connection]:
        """Borrow a connection from the pool, committing on success.

        The connection is rolled back on error and always returned to the pool.
        """
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def save(self, image: ImageObject) -> None:
        # The object key is a uuid-based file name; the image id is that uuid
        # (the key without its extension), recording one row per image.
        image_id = PurePosixPath(image.key).stem
        with self._connection() as conn, conn.cursor() as cur:

            _INSERT_SQL = """
            INSERT INTO image
                (image_id, object_key, bucket, content_type, size_bytes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (image_id) DO NOTHING
            """

            cur.execute(
                _INSERT_SQL,
                (
                    image_id,
                    image.key,
                    image.bucket,
                    image.content_type,
                    image.size_bytes,
                ),
            )

    def close(self) -> None:
        self._pool.closeall()
