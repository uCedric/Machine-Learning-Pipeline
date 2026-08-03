"""Postgres infrastructure shared by every query module in this package.

Mirrors the ``adapters/outbound/db/repository.py`` of the inference and train
services (they are separate Docker build contexts, so the code is copied rather
than imported), with one addition: monitor-service serves its rows straight to
JSON, so the fetch helpers can return ``RealDictCursor`` dicts.

* :class:`Database` owns the process-wide connection pool and the transaction
  contract — borrow a connection, commit on success, roll back on error, always
  return it. The FastAPI lifespan builds **one** and closes it once.
* :class:`PostgresRepository` is the base every query module extends.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from psycopg2.extensions import connection as Connection
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger(__name__)

# Query parameters, as psycopg2 wants them (a sequence bound to %s placeholders).
Params = Sequence[Any]


class Database:
    """The one Postgres connection pool for the process."""

    def __init__(self, dsn: str, minconn: int = 1, maxconn: int = 10) -> None:
        # Accept a SQLAlchemy URL too; psycopg2 wants a plain libpq DSN.
        dsn = dsn.replace("postgresql+psycopg2://", "postgresql://")
        self._pool = ThreadedConnectionPool(minconn, maxconn, dsn)
        logger.info("Postgres connection pool ready (%d-%d connections)", minconn, maxconn)

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        """Borrow a connection for one transaction, committing on success.

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

    def close(self) -> None:
        """Close every pooled connection. Called once, on app shutdown."""
        self._pool.closeall()


class PostgresRepository:
    """Base for the query modules: one :class:`Database`, one fetch helper.

    monitor-service is read-only — the only side effect anywhere in the app is
    publishing ``train`` events to Kafka — so this base exposes no write path.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def _fetch_all(self, sql: str, params: Params = ()) -> list[dict[str, Any]]:
        """Run a read-only query and return rows as dicts, ready to serialise."""
        with self._db.transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                return cur.fetchall()
