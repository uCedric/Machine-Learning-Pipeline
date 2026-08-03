"""Postgres infrastructure shared by every table adapter in this package.

Two pieces, so a table adapter carries nothing but its SQL and its row mapping:

* :class:`Database` owns the process-wide connection pool and the transaction
  contract — borrow a connection, commit on success, roll back on error, always
  return it. The composition root builds **one** and closes it once; no adapter
  creates or closes a pool.
* :class:`PostgresRepository` is the base every table adapter extends. It holds
  the :class:`Database` and turns "a SQL string plus parameters" into a
  statement run inside a single transaction.

This mirrors ``inference-service/adapters/outbound/db/repository.py`` rather than
importing it: the two services are separate Docker build contexts with their own
``domain`` models, so neither can reach into the other.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from psycopg2.extensions import connection as Connection
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger(__name__)

# Query parameters, as psycopg2 wants them (a sequence bound to %s placeholders).
Params = Sequence[Any]


class Database:
    """The one Postgres connection pool for the process."""

    def __init__(self, sql_uri: str, minconn: int = 1, maxconn: int = 10) -> None:
        # ``sql_uri`` is a SQLAlchemy URL (``postgresql+psycopg2://``); psycopg2
        # wants a plain libpq DSN, so drop the driver suffix.
        dsn = sql_uri.replace("postgresql+psycopg2://", "postgresql://")
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
        """Close every pooled connection. Called once, by the composition root."""
        self._pool.closeall()


class PostgresRepository:
    """Base for the table adapters: one :class:`Database`, three statement helpers.

    Subclasses declare their SQL and map rows to domain entities; everything
    about connections, transactions and rollback lives here.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def _execute(self, sql: str, params: Params = ()) -> None:
        """Run a write statement inside one transaction."""
        with self._db.transaction() as conn, conn.cursor() as cur:
            cur.execute(sql, params)

    def _fetch_all(self, sql: str, params: Params = ()) -> list[tuple]:
        """Run a query and return every row."""
        with self._db.transaction() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def _fetch_one(self, sql: str, params: Params = ()) -> tuple | None:
        """Run a query and return the first row, or ``None`` if there is none."""
        with self._db.transaction() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()
