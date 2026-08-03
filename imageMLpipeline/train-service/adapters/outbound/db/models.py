"""``inference_model`` table: read and register trained model versions.

Read to discover which model version to retrain, and written to insert a new
``patch + 1`` candidate row once a retrain produces a new memory bank in object
storage. The table is seeded/maintained by the Flyway migrations in
``migrations/`` (see ``V1__init.sql``).

Note this module is named for the *machine-learning model* entity; the domain
entities live in :mod:`domain.models`.
"""
from __future__ import annotations

import logging
from uuid import uuid4

from adapters.outbound.db.repository import PostgresRepository
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

# Highest patch already taken within a (model_type, major, minor) line, valid or
# not, so a retrain candidate never reuses a version number.
_MAX_PATCH_SQL = """
SELECT MAX(patch_version)
FROM inference_model
WHERE model_type = %s AND major_version = %s AND minor_version = %s
"""

_INSERT_SQL = """
INSERT INTO inference_model
    (model_id, model_type, bucket, major_version, minor_version, patch_version, is_valid)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (model_id) DO NOTHING
"""


class PostgresModelRegistry(PostgresRepository, ModelRegistry):
    """Reads and registers model versions in the Postgres ``inference_model`` table."""

    def latest(self, model_type: str) -> ModelVersion:
        row = self._fetch_one(_LATEST_SQL, (model_type,))
        if row is None:
            raise LookupError(f"No valid model version for model_type '{model_type}'")
        model_id, mtype, bucket, major, minor, patch = row
        version = ModelVersion(
            model_id=str(model_id),
            model_type=mtype,
            bucket=bucket,
            major=major,
            minor=minor,
            patch=patch,
        )
        logger.info(
            "Resolved latest '%s' -> model_id=%s v%s (bucket=%s)",
            model_type,
            version.model_id,
            version.version,
            version.bucket,
        )
        return version

    def next_version(self, base: ModelVersion) -> ModelVersion:
        (max_patch,) = self._fetch_one(
            _MAX_PATCH_SQL, (base.model_type, base.major, base.minor)
        )
        new_patch = (max_patch if max_patch is not None else base.patch) + 1
        version = ModelVersion(
            model_id=str(uuid4()),
            model_type=base.model_type,
            bucket=base.bucket,
            major=base.major,
            minor=base.minor,
            patch=new_patch,
        )
        logger.info(
            "Next version for '%s': v%s (model_id=%s)",
            base.model_type,
            version.version,
            version.model_id,
        )
        return version

    def register(self, version: ModelVersion, *, is_valid: bool) -> None:
        self._execute(
            _INSERT_SQL,
            (
                version.model_id,
                version.model_type,
                version.bucket,
                version.major,
                version.minor,
                version.patch,
                is_valid,
            ),
        )
        logger.info(
            "Registered '%s' v%s (model_id=%s, is_valid=%s)",
            version.model_type,
            version.version,
            version.model_id,
            is_valid,
        )
