"""Postgres adapter: store inference results in a relational table.

Complements the Iceberg sink — the same :class:`~domain.models.InferenceResult`
rows are written to a plain ``inference_results`` table so they are directly
SQL-queryable (Iceberg keeps the data as Parquet in the MinIO warehouse).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import PurePosixPath
from typing import Iterator

from psycopg2.extensions import connection as Connection
from psycopg2.pool import ThreadedConnectionPool

from application.ports.repository import ResultRepository
from domain.models import InferenceResult

logger = logging.getLogger(__name__)


class PostgresResultRepository(ResultRepository):
    """Stores :class:`~domain.models.InferenceResult` rows in Postgres.

    The ``inference_results`` table is provisioned by the Flyway migrations in
    ``migrations/`` (see ``V1__init.sql``), so it must exist before the adapter
    is used.
    """

    def __init__(self, sql_uri: str, minconn: int = 1, maxconn: int = 10) -> None:
        # ``sql_uri`` is a SQLAlchemy URL (``postgresql+psycopg2://``); psycopg2
        # wants a plain libpq DSN, so drop the driver suffix.
        dsn = sql_uri.replace("postgresql+psycopg2://", "postgresql://")
        self._pool = ThreadedConnectionPool(minconn, maxconn, dsn)
        logger.info("Postgres connection pool ready'")

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

    def save(self, result: InferenceResult) -> None:
        # The object key is a uuid-based file name; the event id is that uuid
        # (the key without its extension), recording one execution per image.
        event_id = PurePosixPath(result.object_key).stem
        with self._connection() as conn, conn.cursor() as cur:

            _INSERT_SQL = """
            INSERT INTO inference_results
                (event_id, object_key, bucket, anomaly_score, is_anomaly,
                model_name, heatmap_key, inferred_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (event_id) DO NOTHING
            """

            cur.execute(
                _INSERT_SQL,
                (
                    event_id,
                    result.object_key,
                    result.bucket,
                    result.anomaly_score,
                    result.is_anomaly,
                    result.model_name,
                    result.heatmap_key,
                    result.inferred_at,
                ),
            )

    def close(self) -> None:
        self._pool.closeall()
