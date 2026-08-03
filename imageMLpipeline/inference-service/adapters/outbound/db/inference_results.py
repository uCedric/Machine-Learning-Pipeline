"""``inference_results`` table: stage-one verdicts, written and read back.

Two adapters share this table, so they share this module:

* :class:`PostgresResultRepository` — stage-one **writes** one row per inference
  execution (score, verdict, heatmap location).
* :class:`PostgresClusteringImageSource` — stage-two **reads** those rows back as
  two cohorts: the defect set to cluster and the known-good baseline its
  descriptors are measured against.

The table is provisioned by the Flyway migrations in ``migrations/`` (see
``V1__init.sql``), so it must exist before either adapter is used.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import PurePosixPath

from adapters.outbound.db.repository import PostgresRepository
from application.ports.clustering_image_source import ClusteringImageSource
from application.ports.repository import ResultRepository
from domain.models import InferenceResult

logger = logging.getLogger(__name__)

_INSERT_SQL = """
INSERT INTO inference_results
    (event_id, image_id, model_id, object_key, bucket,
    anomaly_score, status, heatmap_key, inferred_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (event_id) DO NOTHING
"""

# Newest distinct confirmed-defect images, newest first. ``anomaly`` only —
# ``pending`` is the buffer-zone's "uncertain, worth a human look" verdict, and it
# has its own route: an operator reviews it in the monitor dashboard and sends it
# for retraining. Feeding unconfirmed images into an unsupervised clustering that
# is meant to name defect *types* would blur those types.
#
# The inner DISTINCT ON keeps one row per image (its latest verdict); the outer
# query orders those by recency and caps the batch.
_ANOMALOUS_SQL = """
SELECT bucket, object_key FROM (
    SELECT DISTINCT ON (object_key) bucket, object_key, inferred_at
    FROM inference_results
    WHERE status = 'anomaly'
    ORDER BY object_key, inferred_at DESC
) latest
ORDER BY inferred_at DESC
LIMIT %s
"""

# Same as above but bounded below by a watermark: the defects an *assign* run has
# to label, i.e. the ones that arrived after the last completed run.
_ANOMALOUS_SINCE_SQL = """
SELECT bucket, object_key FROM (
    SELECT DISTINCT ON (object_key) bucket, object_key, inferred_at
    FROM inference_results
    WHERE status = 'anomaly'
      AND inferred_at > COALESCE(%s, '-infinity'::timestamptz)
    ORDER BY object_key, inferred_at DESC
) latest
ORDER BY inferred_at DESC
LIMIT %s
"""

# The accumulation gate: how many confirmed-defect rows landed after the last
# completed clustering run, and the newest of their timestamps. Same ``anomaly``
# only rule as above — a run is worth doing once enough *confirmed* defects have
# arrived, so uncertain images must not push the counter towards it.
#
# COALESCE handles the first-ever run, where there is no watermark yet. Counts
# rows rather than distinct images — the gate measures new evidence, and the
# clustering step dedupes by object_key itself.
_BACKLOG_SQL = """
SELECT count(*), max(inferred_at)
FROM inference_results
WHERE status = 'anomaly'
  AND inferred_at > COALESCE(%s, '-infinity'::timestamptz)
"""

# Newest distinct known-good images for a model type, newest first.
_RECENT_GOOD_SQL = """
SELECT bucket, object_key FROM (
    SELECT DISTINCT ON (r.object_key) r.bucket, r.object_key, r.inferred_at
    FROM inference_results r
    JOIN inference_model m ON r.model_id = m.model_id
    WHERE r.status = 'normal' AND m.model_type = %s
    ORDER BY r.object_key, r.inferred_at DESC
) latest
ORDER BY inferred_at DESC
LIMIT %s
"""


class PostgresResultRepository(PostgresRepository, ResultRepository):
    """Stores :class:`~domain.models.InferenceResult` rows in Postgres."""

    def save(self, result: InferenceResult) -> None:
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
                result.anomaly_score,
                result.status.value,
                result.heatmap_key,
                result.inferred_at,
            ),
        )


class PostgresClusteringImageSource(PostgresRepository, ClusteringImageSource):
    """Reads the defect / known-good cohorts a stage-two batch run needs.

    A defect image is the newest row per ``object_key`` whose status is
    ``anomaly``; a known-good image is the newest ``normal`` row
    for the stage-one model type. Both dedupe by ``object_key`` so a re-inferred
    image is clustered once.
    """

    def anomalous(self, limit: int) -> list[tuple[str, str]]:
        rows = self._fetch_all(_ANOMALOUS_SQL, (limit,))
        logger.info("Defect set: %d anomaly image(s) (limit %d)", len(rows), limit)
        return [(bucket, key) for bucket, key in rows]

    def recent_good(self, model_type: str, limit: int) -> list[tuple[str, str]]:
        rows = self._fetch_all(_RECENT_GOOD_SQL, (model_type, limit))
        logger.info(
            "Normal baseline: %d known-good image(s) for '%s' (limit %d)",
            len(rows),
            model_type,
            limit,
        )
        return [(bucket, key) for bucket, key in rows]

    def anomalous_since(
        self, since: datetime | None, limit: int
    ) -> list[tuple[str, str]]:
        rows = self._fetch_all(_ANOMALOUS_SINCE_SQL, (since, limit))
        logger.info(
            "New defect set: %d anomaly image(s) since %s (limit %d)",
            len(rows),
            since.isoformat() if since else "the beginning",
            limit,
        )
        return [(bucket, key) for bucket, key in rows]

    def backlog_since(self, since: datetime | None) -> tuple[int, datetime | None]:
        rows = self._fetch_all(_BACKLOG_SQL, (since,))
        count, newest = rows[0] if rows else (0, None)
        return int(count), newest
