"""``stage_two_run`` table: when defect clustering last ran, and how far it got.

One adapter, :class:`PostgresClusteringRunLog`, implementing
:class:`~application.ports.clustering_runs.ClusteringRunLog`. Stage-two reads the
watermark here to decide whether enough new defect images have accumulated to be
worth a run, and appends a row when the run finishes.

The table is provisioned by the Flyway migrations in ``migrations/`` (see
``V4__add_stage_two_runs.sql``), so it must exist before the adapter is used.
"""
from __future__ import annotations

import logging
from datetime import datetime

from adapters.outbound.db.repository import PostgresRepository
from application.ports.clustering_runs import ClusteringRunLog
from domain.models import ClusteringRun

logger = logging.getLogger(__name__)

# Only completed runs move the watermark: a failed run persisted nothing, so its
# backlog must stay pending. The partial index on (covered_through DESC) makes
# this a single index read.
_WATERMARK_SQL = """
SELECT max(covered_through)
FROM stage_two_run
WHERE status = 'completed'
  AND (%s IS NULL OR mode = %s::stage_two_run_mode)
"""

_INSERT_SQL = """
INSERT INTO stage_two_run
    (run_id, model_id, mode, state_model_id, status, covered_through,
     defect_count, cluster_count, labelled_count, noise_count, mean_probability,
     error, started_at, finished_at)
VALUES (%s, %s, %s::stage_two_run_mode, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (run_id) DO NOTHING
"""


class PostgresClusteringRunLog(PostgresRepository, ClusteringRunLog):
    """Stores :class:`~domain.models.ClusteringRun` rows in Postgres."""

    def last_watermark(self, mode: str | None = None) -> datetime | None:
        rows = self._fetch_all(_WATERMARK_SQL, (mode, mode))
        return rows[0][0] if rows else None

    def record(self, run: ClusteringRun) -> None:
        self._execute(
            _INSERT_SQL,
            (
                run.run_id,
                run.model_id,
                run.mode,
                run.state_model_id,
                run.status,
                run.covered_through,
                run.defect_count,
                run.cluster_count,
                run.labelled_count,
                run.noise_count,
                run.mean_probability,
                run.error,
                run.started_at,
                run.finished_at,
            ),
        )
        logger.info(
            "Recorded stage-two %s run %s: %s, %d defect image(s), watermark=%s",
            run.mode,
            run.run_id,
            run.status,
            run.defect_count,
            run.covered_through,
        )
