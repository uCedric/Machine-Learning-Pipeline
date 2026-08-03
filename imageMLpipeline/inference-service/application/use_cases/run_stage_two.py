"""Stage-two use case: cluster the defect set, or assign new defects to it.

Pure application logic wired entirely against :mod:`application.ports`: the batch
clustering model, the clustering image source, object storage (as plain callables),
the cluster-result repository, the run log and the model registry. It holds no
reference to torch, UMAP, HDBSCAN, MinIO or Postgres — the composition root
injects concrete adapters.

A ``stage-two`` event is only a *trigger*, and most triggers do nothing. What a
trigger can turn into is one of two runs, chosen by how much has accumulated:

* **assign** (``assign_count`` new defects — cheap) — project only the newly
  arrived defects through the frozen fit of the current model version. They take
  ids from clusters that already exist, so a ``cluster_id`` keeps meaning the same
  defect type it did in every earlier batch.
* **fit** (``refit_count`` new defects since the last fit — expensive) — cluster
  the whole defect set from scratch, then freeze the fitted pipeline as a **new
  model version**: artifacts to object storage, one row in the registry. A fit
  renumbers the clusters, which is why it is a version boundary and why
  ``cluster_results.model_id`` is what distinguishes two numbering schemes.

The two thresholds need two watermarks, both derived from the run log rather than
counted in memory: the assign gate counts from the last run of *any* mode, the fit
gate from the last **fit**. Sharing one would reset the fit count on every assign,
so the clusters would never be redrawn.

Bootstrap and fallback both collapse to "fit": if the current version carries no
frozen state, or the model cannot be frozen at all, a fit is the only option.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from application.ports.cluster_repository import ClusterResultRepository
from application.ports.clustering_image_source import ClusteringImageSource
from application.ports.clustering_runs import ClusteringRunLog
from application.ports.model import ClusterModel
from application.ports.model_registry import ModelRegistry
from domain.models import ClusteringRun, ClusterResult, InferenceEvent

logger = logging.getLogger(__name__)

# Images are fetched from object storage concurrently before clustering; bound
# the number of open connections.
_FETCH_WORKERS = 16


class RunStageTwo:
    def __init__(
        self,
        cluster_model: ClusterModel,
        image_source: ClusteringImageSource,
        cluster_repository: ClusterResultRepository,
        run_log: ClusteringRunLog,
        registry: ModelRegistry,
        *,
        storage_get,
        state_put,
        model_id: str,
        model_key: str,
        normal_model_type: str,
        min_batch: int,
        refit_count: int,
        assign_count: int,
        normal_set_size: int,
        defect_set_size: int,
    ) -> None:
        self._cluster_model = cluster_model
        self._image_source = image_source
        self._cluster_repository = cluster_repository
        self._run_log = run_log
        self._registry = registry
        # Object storage as plain callables so the use case stays free of the
        # adapter's type: read image bytes, and write a frozen fit for a version.
        self._storage_get = storage_get
        self._state_put = state_put
        self._model_id = model_id
        # The registry key to bump when freezing a fit (e.g. 'resnet50').
        self._model_key = model_key
        self._normal_model_type = normal_model_type
        self._min_batch = min_batch
        self._refit_count = refit_count
        self._assign_count = assign_count
        self._normal_set_size = normal_set_size
        self._defect_set_size = defect_set_size

    # ---- the gate -----------------------------------------------------------
    def execute(self, event: InferenceEvent) -> None:
        fit_watermark = self._run_log.last_watermark("fit")
        since_fit, newest_fit = self._image_source.backlog_since(fit_watermark)

        # The fit gate wins when both are open: a re-fit supersedes an assign,
        # since it relabels the whole set anyway.
        if since_fit >= max(1, self._refit_count) and newest_fit is not None:
            self._run("fit", newest_fit, since_fit)
            return

        if not self._can_assign:
            # No frozen state to assign against (a freshly seeded version, or a
            # model whose pipeline was never split), so the fit gate is the only
            # gate there is.
            self._log_waiting("fit", since_fit, self._refit_count, fit_watermark)
            return

        run_watermark = self._run_log.last_watermark()
        since_run, newest_run = self._image_source.backlog_since(run_watermark)
        if since_run >= max(1, self._assign_count) and newest_run is not None:
            self._run("assign", newest_run, since_run, watermark=run_watermark)
            return
        self._log_waiting("assign", since_run, self._assign_count, run_watermark)

    @property
    def _can_assign(self) -> bool:
        """Whether the loaded model version carries a frozen fit to assign against."""
        return getattr(self._cluster_model, "state", None) is not None

    @staticmethod
    def _log_waiting(mode: str, have: int, want: int, watermark: datetime | None) -> None:
        logger.info(
            "Stage-two %s gate: %d/%d new defect image(s) since %s; waiting",
            mode,
            have,
            want,
            watermark.isoformat() if watermark else "the beginning",
        )

    def _run(
        self,
        mode: str,
        covered_through: datetime,
        new_defects: int,
        watermark: datetime | None = None,
    ) -> None:
        """Execute one run, recording it either way.

        ``covered_through`` was sampled *before* the work starts, so defects that
        land while a run is in flight count towards the next run instead of being
        skipped by this one.
        """
        logger.info(
            "Stage-two %s run starting: %d new defect image(s)", mode, new_defects
        )
        started_at = datetime.now(timezone.utc)
        try:
            if mode == "fit":
                self._do_fit(covered_through, new_defects, started_at)
            else:
                self._do_assign(covered_through, new_defects, watermark, started_at)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            # Record the failure so it is visible, but leave the watermark where it
            # was: that backlog is still unconsumed and must be retried.
            self._run_log.record(
                ClusteringRun(
                    model_id=self._model_id,
                    covered_through=covered_through,
                    defect_count=new_defects,
                    mode=mode,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    started_at=started_at,
                )
            )
            raise

    # ---- fit ----------------------------------------------------------------
    def _do_fit(
        self, covered_through: datetime, new_defects: int, started_at: datetime
    ) -> None:
        defect_locs = self._image_source.anomalous(self._defect_set_size)
        defect_locs, defect_bytes = self._fetch(defect_locs)
        if len(defect_bytes) < self._min_batch:
            logger.warning(
                "Fit gate opened but only %d defect image(s) could be fetched "
                "(< min batch %d); not clustering, watermark unchanged",
                len(defect_bytes),
                self._min_batch,
            )
            return

        normal_locs = self._image_source.recent_good(
            self._normal_model_type, self._normal_set_size
        )
        _, normal_bytes = self._fetch(normal_locs)

        logger.info(
            "Fitting %d defect image(s) against %d known-good image(s)",
            len(defect_bytes),
            len(normal_bytes),
        )
        fit = self._cluster_model.fit(defect_bytes, normal_bytes)

        # Freeze the fit as a new version before writing results, so the rows are
        # attributed to the version that produced them. Artifacts first, registry
        # row second: a row pointing at a prefix that does not exist yet would
        # break every later load.
        state_version = None
        if fit.state is not None:
            state_version = self._registry.next_minor(self._model_key)
            self._state_put(state_version, fit.state)
            self._registry.register(state_version)
            # The clusters are now numbered by the new version, so every later
            # assign must be attributed to it — including the ids the model itself
            # stamps onto its assignments.
            self._cluster_model.model_id = state_version.model_id
            self._model_id = state_version.model_id
            logger.info(
                "Froze the fit as '%s' v%s (%d bytes); later batches assign to it",
                state_version.model_type,
                state_version.version,
                len(fit.state),
            )
        else:
            logger.warning(
                "Model produced no reusable state; every stage-two run will re-fit"
            )

        attributed_to = state_version.model_id if state_version else self._model_id
        run_ts = self._save(defect_locs, fit.assignments, attributed_to)

        self._run_log.record(
            ClusteringRun(
                model_id=self._model_id,
                covered_through=covered_through,
                defect_count=new_defects,
                mode="fit",
                state_model_id=state_version.model_id if state_version else None,
                started_at=started_at,
                finished_at=run_ts,
                **self._quality(fit.assignments),
            )
        )

    # ---- assign -------------------------------------------------------------
    def _do_assign(
        self,
        covered_through: datetime,
        new_defects: int,
        watermark: datetime | None,
        started_at: datetime,
    ) -> None:
        defect_locs = self._image_source.anomalous_since(
            watermark, self._defect_set_size
        )
        defect_locs, defect_bytes = self._fetch(defect_locs)
        if not defect_bytes:
            logger.warning(
                "Assign gate opened but no new defect image could be fetched; "
                "watermark unchanged"
            )
            return

        logger.info(
            "Assigning %d new defect image(s) to the frozen fit", len(defect_bytes)
        )
        assignments = self._cluster_model.assign(defect_bytes)
        run_ts = self._save(defect_locs, assignments, self._model_id)

        quality = self._quality(assignments)
        if quality["noise_count"]:
            logger.info(
                "%d of %d assigned image(s) matched no cluster (%.0f%%); a sustained "
                "rise here is feature drift — the monitor dashboard surfaces it so an "
                "operator can decide whether the clustering algorithm needs rework",
                quality["noise_count"],
                quality["labelled_count"],
                100 * quality["noise_count"] / quality["labelled_count"],
            )
        self._run_log.record(
            ClusteringRun(
                model_id=self._model_id,
                covered_through=covered_through,
                defect_count=new_defects,
                mode="assign",
                state_model_id=self._model_id,
                started_at=started_at,
                finished_at=run_ts,
                **quality,
            )
        )

    # ---- shared -------------------------------------------------------------
    @staticmethod
    def _quality(assignments) -> dict:
        """Per-run quality figures, the inputs to the dashboard's drift view.

        ``noise_count / labelled_count`` is the headline: an assign can only place
        an image into a cluster that already exists, so a genuinely new defect type
        comes back as -1. ``mean_probability`` catches the same drift earlier —
        points landing at the edge of a cluster rather than in it.
        """
        labelled = len(assignments)
        return {
            "labelled_count": labelled,
            "noise_count": sum(1 for a in assignments if a.cluster_id == -1),
            "cluster_count": len(
                {a.cluster_id for a in assignments if a.cluster_id != -1}
            ),
            "mean_probability": (
                sum(a.probability for a in assignments) / labelled if labelled else None
            ),
        }

    def _save(self, locations, assignments, model_id: str) -> datetime:
        """Persist one row per image, all sharing this run's timestamp."""
        run_ts = datetime.now(timezone.utc)
        saved = 0
        for (bucket, key), assignment in zip(locations, assignments):
            self._cluster_repository.save(
                ClusterResult(
                    bucket=bucket,
                    object_key=key,
                    cluster_id=assignment.cluster_id,
                    probability=assignment.probability,
                    model_id=model_id,
                    clustered_at=run_ts,
                )
            )
            saved += 1
        logger.info("Saved %d cluster result(s) for model_id=%s", saved, model_id)
        return run_ts

    def _fetch(
        self, locations: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], list[bytes]]:
        """Fetch image bytes for ``locations`` concurrently, preserving order.

        A location whose fetch fails is logged and dropped so the kept locations
        and bytes stay index-aligned for the clustering step.
        """
        if not locations:
            return [], []
        with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
            results = list(pool.map(lambda loc: self._safe_get(*loc), locations))
        kept_locs, kept_bytes = [], []
        for loc, data in zip(locations, results):
            if data is None:
                continue
            kept_locs.append(loc)
            kept_bytes.append(data)
        return kept_locs, kept_bytes

    def _safe_get(self, bucket: str, key: str) -> bytes | None:
        try:
            return self._storage_get(bucket, key)
        except Exception:
            logger.warning("Failed to fetch image %s/%s; skipping", bucket, key)
            return None
