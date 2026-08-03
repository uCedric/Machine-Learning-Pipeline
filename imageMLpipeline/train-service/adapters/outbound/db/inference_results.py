"""``inference_results`` table: the rolling validation set of known-good images.

Reads the most recent images a prior inference classified as ``normal``. These
known-good images are re-scored against a freshly-updated memory bank to
recalibrate the anomaly-score buffer zone (see
:class:`application.use_cases.run_training.RunTraining`).

Despite implementing the :class:`~application.ports.image_repository.ImageRepository`
port, this adapter never touches the ``image`` table — it lives here because
``inference_results`` is the table it reads. The table is provisioned by the
Flyway migrations in ``migrations/`` (see ``V1__init.sql``).
"""
from __future__ import annotations

import logging

from adapters.outbound.db.repository import PostgresRepository
from application.ports.image_repository import ImageRepository

logger = logging.getLogger(__name__)

# Most recent images classified ``normal`` for a model type, newest first.
_RECENT_GOOD_SQL = """
SELECT r.bucket, r.object_key
FROM inference_results r
JOIN inference_model m ON r.model_id = m.model_id
WHERE r.status = 'normal' AND m.model_type = %s
ORDER BY r.inferred_at DESC
LIMIT %s
"""


class PostgresImageRepository(PostgresRepository, ImageRepository):
    """Reads recent known-good images from the Postgres ``inference_results`` table."""

    def recent_good(self, model_type: str, limit: int) -> list[tuple[str, str]]:
        rows = self._fetch_all(_RECENT_GOOD_SQL, (model_type, limit))
        logger.info(
            "Validation set: %d known-good image(s) for '%s' (limit %d)",
            len(rows),
            model_type,
            limit,
        )
        return [(bucket, key) for bucket, key in rows]
