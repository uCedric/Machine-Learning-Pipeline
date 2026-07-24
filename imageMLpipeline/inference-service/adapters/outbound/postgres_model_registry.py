"""Postgres adapter: resolve registered model versions from ``inference_model``.

Reads the model registry seeded/maintained by the Flyway migrations in
``migrations/`` (see ``V1__init.sql``). Used at start-up to discover which model
version to load and where its assets live in object storage.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from psycopg2.extensions import connection as Connection
from psycopg2.pool import ThreadedConnectionPool

from application.ports.model_registry import ModelRegistry
from domain.models import ModelVersion

logger = logging.getLogger(__name__)

# Newest valid version for a model type: highest semver wins.
_LATEST_SQL = """
SELECT model_id, model_type, bucket, major_version, minor_version, patch_version
FROM inference_model
WHERE model_type = %s AND is_valid = true
ORDER BY major_version DESC, minor_version DESC, patch_version DESC
LIMIT 1
"""


class PostgresModelRegistry(ModelRegistry):
    """Resolves model versions from the Postgres ``inference_model`` table."""

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

    def latest(self, model_type: str) -> ModelVersion:
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(_LATEST_SQL, (model_type,))
            row = cur.fetchone()
        if row is None:
            raise LookupError(f"No valid model version for model_type '{model_type}'")
        model_id, mtype, bucket, major, minor, patch = row
        semver = f"{major}.{minor}.{patch}"
        version = ModelVersion(
            model_id=str(model_id),
            model_type=mtype,
            bucket=bucket,
            version=semver,
        )
        logger.info(
            "Resolved latest '%s' -> model_id=%s v%s (bucket=%s)",
            model_type,
            version.model_id,
            version.version,
            version.bucket,
        )
        return version

    def close(self) -> None:
        self._pool.closeall()
