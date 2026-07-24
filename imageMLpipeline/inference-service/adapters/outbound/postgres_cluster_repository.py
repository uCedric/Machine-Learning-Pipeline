"""Postgres adapter: store cluster results in a relational table.

:class:`~domain.models.ClusterResult` rows are written to a plain
``cluster_results`` table so they are directly SQL-queryable.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import PurePosixPath
from typing import Iterator

from psycopg2.extensions import connection as Connection
from psycopg2.pool import ThreadedConnectionPool

from application.ports.cluster_repository import ClusterResultRepository
from domain.models import ClusterResult

logger = logging.getLogger(__name__)


class PostgresClusterResultRepository(ClusterResultRepository):
    """Stores :class:`~domain.models.ClusterResult` rows in Postgres.

    The ``cluster_results`` table is provisioned by the Flyway migrations in
    ``migrations/`` (see ``V2__add_cluster_results.sql``), so it must exist
    before the adapter is used.
    """

    def __init__(self, sql_uri: str, minconn: int = 1, maxconn: int = 10) -> None:
        # ``sql_uri`` is a SQLAlchemy URL (``postgresql+psycopg2://``); psycopg2
        # wants a plain libpq DSN, so drop the driver suffix.
        dsn = sql_uri.replace("postgresql+psycopg2://", "postgresql://")
        self._pool = ThreadedConnectionPool(minconn, maxconn, dsn)
        logger.info("Postgres connection pool ready")

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

    def save(self, result: ClusterResult) -> None:
        # ``event_id`` is the result's own id. ``image_id`` (FK → image) is the
        # uuid stem of the object key, matching the row the ELT recorded.
        image_id = PurePosixPath(result.object_key).stem
        with self._connection() as conn, conn.cursor() as cur:

            _INSERT_SQL = """
            INSERT INTO cluster_results
                (event_id, image_id, model_id, object_key, bucket,
                cluster_id, probability, clustered_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (event_id) DO NOTHING
            """

            cur.execute(
                _INSERT_SQL,
                (
                    result.event_id,
                    image_id,
                    result.model_id,
                    result.object_key,
                    result.bucket,
                    result.cluster_id,
                    result.probability,
                    result.clustered_at,
                ),
            )

    def close(self) -> None:
        self._pool.closeall()
