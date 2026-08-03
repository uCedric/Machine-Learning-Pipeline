"""``stage_two_run`` table: read-only views of stage-two clustering health.

Stage-two either **fits** (redraws the clusters and freezes a new model version)
or **assigns** (labels newly arrived defects against that frozen fit). An assign
can only place an image into a cluster that already exists, so a defect type the
fit never saw comes back as ``cluster_id = -1``.

That makes the noise ratio the drift signal, and this module exists to surface it:
the pipeline cannot repair a clustering whose feature space has moved, so the
decision to rework the algorithm is a human one, and the dashboard's job is to
make the evidence obvious rather than to act on it.

Two views:

* :meth:`recent` — the run history, one row per run, with the ratios precomputed.
* :meth:`health` — the current verdict: the newest fit as a baseline, the assigns
  since it, and whether their noise has drifted away from what the fit achieved.
"""
from __future__ import annotations

import logging
from typing import Any

from db.repository import PostgresRepository

logger = logging.getLogger(__name__)

# Ratios are computed in SQL so the API and the template never divide by zero, and
# so a run that labelled nothing (a failure) reports NULL rather than 0%.
_METRICS = """
    r.run_id, r.mode::text AS mode, r.status::text AS status,
    r.defect_count, r.labelled_count, r.cluster_count,
    r.noise_count, r.mean_probability, r.error,
    r.started_at, r.finished_at,
    r.state_model_id,
    round(
        r.noise_count::numeric / NULLIF(r.labelled_count, 0), 4
    )::float AS noise_ratio,
    EXTRACT(EPOCH FROM (r.finished_at - r.started_at))::float AS duration_s,
    m.major_version || '.' || m.minor_version || '.' || m.patch_version AS version
"""

_RECENT_SQL = f"""
SELECT {_METRICS}
FROM stage_two_run r
LEFT JOIN inference_model m ON r.state_model_id = m.model_id
ORDER BY r.started_at DESC
LIMIT %s
"""

# The newest completed fit is the baseline: the noise ratio it achieved on the very
# data it was built from is the best case this model version can do. Assign runs
# against that same version are then compared with it.
_BASELINE_SQL = """
SELECT run_id, state_model_id, labelled_count, noise_count, mean_probability,
       started_at,
       round(noise_count::numeric / NULLIF(labelled_count, 0), 4)::float AS noise_ratio
FROM stage_two_run
WHERE mode = 'fit' AND status = 'completed' AND labelled_count > 0
ORDER BY started_at DESC
LIMIT 1
"""

# Assign runs that used a given frozen state, newest first — the trend to plot.
_ASSIGNS_SINCE_SQL = """
SELECT run_id, labelled_count, noise_count, mean_probability, started_at,
       round(noise_count::numeric / NULLIF(labelled_count, 0), 4)::float AS noise_ratio
FROM stage_two_run
WHERE mode = 'assign' AND status = 'completed' AND labelled_count > 0
  AND state_model_id = %s
ORDER BY started_at DESC
LIMIT %s
"""

# How much has accumulated towards the next gate, so the dashboard can say "487/500"
# rather than leaving an operator guessing why nothing is happening.
_PENDING_SQL = """
SELECT
    (SELECT count(*) FROM inference_results
      WHERE status = 'anomaly'
        AND inferred_at > COALESCE(
            (SELECT max(covered_through) FROM stage_two_run WHERE status = 'completed'),
            '-infinity'::timestamptz)) AS since_any_run,
    (SELECT count(*) FROM inference_results
      WHERE status = 'anomaly'
        AND inferred_at > COALESCE(
            (SELECT max(covered_through) FROM stage_two_run
              WHERE status = 'completed' AND mode = 'fit'),
            '-infinity'::timestamptz)) AS since_last_fit
"""


class StageTwoRunQueries(PostgresRepository):
    """Read-only queries over ``stage_two_run`` for the dashboard."""

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._fetch_all(_RECENT_SQL, (limit,))

    def pending(self) -> dict[str, Any]:
        rows = self._fetch_all(_PENDING_SQL, ())
        return rows[0] if rows else {"since_any_run": 0, "since_last_fit": 0}

    def health(self, drift_ratio: float, assign_window: int = 10) -> dict[str, Any]:
        """Is the current clustering still describing the incoming defects?

        ``drift_ratio`` is the absolute noise share that counts as "bad" on its own.
        A run is also flagged when its noise is more than twice the baseline the fit
        achieved, which catches a model that was always noisy drifting further, and
        avoids crying drift over a fit that legitimately leaves 25% as noise.

        Returns ``verdict`` of ``unknown`` (nothing fitted yet), ``healthy``, or
        ``drifting`` — plus the numbers behind it, because the operator, not this
        function, decides whether the algorithm needs rework.
        """
        baseline_rows = self._fetch_all(_BASELINE_SQL, ())
        pending = self.pending()
        if not baseline_rows:
            return {
                "verdict": "unknown",
                "reason": "no completed fit yet; the first run will be a fit",
                "baseline": None,
                "assigns": [],
                "pending": pending,
            }

        baseline = baseline_rows[0]
        assigns = self._fetch_all(
            _ASSIGNS_SINCE_SQL, (baseline["state_model_id"], assign_window)
        )
        base_ratio = baseline["noise_ratio"] or 0.0
        limit = max(drift_ratio, base_ratio * 2)

        drifted = [a for a in assigns if (a["noise_ratio"] or 0.0) >= limit]
        verdict = "drifting" if drifted else "healthy"
        reason = (
            f"{len(drifted)} of the last {len(assigns)} assign run(s) exceeded "
            f"{limit:.0%} unmatched (fit baseline {base_ratio:.0%})"
            if drifted
            else f"last {len(assigns)} assign run(s) within {limit:.0%} unmatched "
            f"(fit baseline {base_ratio:.0%})"
        )
        if verdict == "drifting":
            logger.warning("Stage-two clustering looks to be drifting: %s", reason)
        return {
            "verdict": verdict,
            "reason": reason,
            "threshold": limit,
            "baseline": baseline,
            "assigns": assigns,
            "pending": pending,
        }
