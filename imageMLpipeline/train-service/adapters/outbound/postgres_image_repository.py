"""Postgres adapter: the rolling validation set of known-good images.

Reads the most recent images a prior inference classified as ``normal`` from the
``inference_results`` table (provisioned by the Flyway migrations in
``migrations/``; see ``V1__init.sql``). These known-good images are re-scored
against a freshly-updated memory bank to recalibrate the anomaly-score buffer
zone (see :class:`application.use_cases.run_training.RunTraining`).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from psycopg2.extensions import connection as Connection
from psycopg2.pool import ThreadedConnectionPool

from application.ports.image_repository import ImageRepository

logger = logging.getLogger(__name__)

# Most recent images classified ``normal`` for a model type, newest first.
_RECENT_GOOD_SQL = """
SELECT r.bucket, r.object_key
FROM inference_results r
JOIN inference_model m ON r.model_id = m.model_id
WHERE r.status = 'normal' AND m.model_type = %s
ORDER BY r.inferred_at DESC
LIMIT %s
"""


class PostgresImageRepository(ImageRepository):
    """Reads recent known-good images from the Postgres ``inference_results`` table."""

    def __init__(self, sql_uri: str, minconn: int = 1, maxconn: int = 2) -> None:
        # ``sql_uri`` is a SQLAlchemy URL (``postgresql+psycopg2://``); psycopg2
        # wants a plain libpq DSN, so drop the driver suffix.
        dsn = sql_uri.replace("postgresql+psycopg2://", "postgresql://")
        self._pool = ThreadedConnectionPool(minconn, maxconn, dsn)

    @contextmanager
    def _connection(self) -> Iterator[Connection]:
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def recent_good(self, model_type: str, limit: int) -> list[tuple[str, str]]:
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(_RECENT_GOOD_SQL, (model_type, limit))
            rows = cur.fetchall()
        logger.info(
            "Validation set: %d known-good image(s) for '%s' (limit %d)",
            len(rows),
            model_type,
            limit,
        )
        return [(bucket, key) for bucket, key in rows]

    def close(self) -> None:
        self._pool.closeall()
