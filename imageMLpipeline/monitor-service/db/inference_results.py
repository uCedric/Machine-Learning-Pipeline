"""``inference_results`` table: the rows the dashboard reads.

Every statement the HTTP layer needs lives here as a named method, so the
endpoints in ``app.py`` stay about HTTP concerns (validation, status codes,
serialisation) and never carry SQL.
"""
from __future__ import annotations

from typing import Any

from db.repository import PostgresRepository

_COLUMNS = """
    event_id, image_id, object_key, bucket, anomaly_score,
    status, heatmap_key, inferred_at
"""

_RECENT_SQL = f"""
SELECT {_COLUMNS}
FROM inference_results
ORDER BY inferred_at DESC
LIMIT %s
"""

_RECENT_BY_STATUS_SQL = f"""
SELECT {_COLUMNS}
FROM inference_results
WHERE status = %s
ORDER BY inferred_at DESC
LIMIT %s
"""

_STATUS_COUNTS_SQL = """
SELECT status, COUNT(*) AS n
FROM inference_results
GROUP BY status
"""

# Re-validated server-side: only genuinely pending rows are ever republished,
# regardless of what the client sent.
_PENDING_BY_IDS_SQL = """
SELECT r.event_id, r.bucket, r.object_key, m.model_type
FROM inference_results r
JOIN inference_model m ON m.model_id = r.model_id
WHERE r.event_id = ANY(%s::uuid[]) AND r.status = 'pending'
"""


class InferenceResultsQueries(PostgresRepository):
    """Read-only queries over ``inference_results``."""

    def recent(self, limit: int, status: str | None = None) -> list[dict[str, Any]]:
        """Most recent results, newest first, optionally filtered by status."""
        if status:
            return self._fetch_all(_RECENT_BY_STATUS_SQL, (status, limit))
        return self._fetch_all(_RECENT_SQL, (limit,))

    def status_counts(self) -> list[dict[str, Any]]:
        """One ``{status, n}`` row per verdict present in the table."""
        return self._fetch_all(_STATUS_COUNTS_SQL)

    def pending_by_ids(self, event_ids: list[str]) -> list[dict[str, Any]]:
        """The subset of ``event_ids`` that are still ``pending``, with their model type."""
        return self._fetch_all(_PENDING_BY_IDS_SQL, (event_ids,))
