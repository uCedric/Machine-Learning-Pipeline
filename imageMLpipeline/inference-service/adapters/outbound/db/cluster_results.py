"""``cluster_results`` table: stage-two defect-cluster assignments.

Modelled on ``inference_results`` — one row per clustering execution, joinable
back to it on ``image_id``. No row for an image means clustering did not run;
``cluster_id = -1`` is a real verdict (HDBSCAN noise — the image matched no
known cluster).

The table is provisioned by the Flyway migrations in ``migrations/`` (see
``V2__add_cluster_results.sql``), so it must exist before the adapter is used.
"""
from __future__ import annotations

from pathlib import PurePosixPath

from adapters.outbound.db.repository import PostgresRepository
from application.ports.cluster_repository import ClusterResultRepository
from domain.models import ClusterResult

_INSERT_SQL = """
INSERT INTO cluster_results
    (event_id, image_id, model_id, object_key, bucket,
    cluster_id, probability, clustered_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (event_id) DO NOTHING
"""


class PostgresClusterResultRepository(PostgresRepository, ClusterResultRepository):
    """Stores :class:`~domain.models.ClusterResult` rows in Postgres."""

    def save(self, result: ClusterResult) -> None:
        # ``event_id`` is the result's own id. ``image_id`` (FK → image) is the
        # uuid stem of the object key, matching the row the ELT recorded.
        image_id = PurePosixPath(result.object_key).stem
        self._execute(
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
