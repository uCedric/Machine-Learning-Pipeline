"""``inference_model`` table: the registry of trained model versions.

Read at start-up to discover which model version to load and where its assets
live in object storage. The table is seeded/maintained by the Flyway migrations
in ``migrations/`` (see ``V1__init.sql`` for PatchCore and
``V3__add_stage_two_clustering.sql`` for the clustering backbones).

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

# Highest version of a type regardless of validity — a bump must clear *every*
# existing version, or a re-fit could collide with a registered candidate.
_MAX_VERSION_SQL = """
SELECT bucket, major_version, minor_version, patch_version
FROM inference_model
WHERE model_type = %s
ORDER BY major_version DESC, minor_version DESC, patch_version DESC
LIMIT 1
"""

_INSERT_SQL = """
INSERT INTO inference_model
    (model_id, model_type, bucket, major_version, minor_version, patch_version, is_valid)
VALUES (%s, %s, %s, %s, %s, %s, true)
ON CONFLICT (model_id) DO NOTHING
"""


class PostgresModelRegistry(PostgresRepository, ModelRegistry):
    """Resolves and records model versions in the Postgres ``inference_model`` table."""

    def latest(self, model_type: str) -> ModelVersion:
        row = self._fetch_one(_LATEST_SQL, (model_type,))
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

    def next_minor(self, model_type: str) -> ModelVersion:
        """Bump the newest valid version's minor number, keeping bucket and type.

        A new ``model_id`` is minted here (not by the database) so the caller can
        name the artifact prefix before the row exists. Minor rather than patch:
        a re-fit changes the cluster ids, which is a behavioural change for anyone
        reading ``cluster_results``, not a silent fix.
        """
        row = self._fetch_one(_MAX_VERSION_SQL, (model_type,))
        if row is None:
            raise LookupError(f"No model version to bump for model_type '{model_type}'")
        bucket, major, minor, _patch = row
        return ModelVersion(
            model_id=str(uuid4()),
            model_type=model_type,
            bucket=bucket,
            version=f"{major}.{minor + 1}.0",
        )

    def register(self, version: ModelVersion) -> None:
        major, minor, patch = (int(p) for p in version.version.split("."))
        self._execute(
            _INSERT_SQL,
            (
                version.model_id,
                version.model_type,
                version.bucket,
                major,
                minor,
                patch,
            ),
        )
        logger.info(
            "Registered '%s' v%s as valid (model_id=%s)",
            version.model_type,
            version.version,
            version.model_id,
        )
