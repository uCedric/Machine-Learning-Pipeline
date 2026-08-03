-- Flyway V4: stage-two run log — the accumulation gate for defect clustering.
--
-- Stage-one emits one stage-two trigger per anomaly/pending image, but a
-- whole-set clustering per trigger is wasteful: the defect set barely changes
-- between two consecutive images. This table turns those triggers into a gate —
-- clustering runs only once STAGE_TWO_TRIGGER_COUNT *new* defect images have
-- accumulated since the last completed run.
--
-- Why a table rather than deriving the count from cluster_results: a run that
-- produces no rows (every image classed as HDBSCAN noise) or that fails midway
-- would leave max(clustered_at) unchanged, so the same backlog would be
-- re-clustered forever. Recording the run explicitly — with the watermark it
-- covered — makes the gate advance exactly once per completed run.
--
-- The row count itself is never stored: it is always derived from
-- inference_results, which stays the single source of truth. A stage-one replay
-- that re-inserts result rows therefore cannot corrupt a counter, because there
-- is no counter.

-- A run either finished or it did not; a failed run must not advance the
-- watermark, so the gate query filters on this.
CREATE TYPE stage_two_run_status AS ENUM ('completed', 'failed');

-- stage_two_run schema --
CREATE TABLE IF NOT EXISTS stage_two_run (
    run_id          UUID PRIMARY KEY,
    model_id        UUID NOT NULL REFERENCES inference_model(model_id),
    status          stage_two_run_status NOT NULL,
    -- Newest inference_results.inferred_at this run took into account. The next
    -- run counts only defects newer than this, so images that land *during* a
    -- run are picked up by the following one rather than being skipped.
    covered_through TIMESTAMPTZ NOT NULL,
    -- How many defect images went into the clustering, and how many distinct
    -- clusters came out (excluding -1 noise). cluster_count is NULL for a
    -- failed run.
    defect_count    INTEGER NOT NULL,
    cluster_count   INTEGER,
    error           TEXT,
    started_at      TIMESTAMPTZ NOT NULL,
    finished_at     TIMESTAMPTZ NOT NULL
);

-- The gate reads one row per trigger: the newest completed watermark.
CREATE INDEX IF NOT EXISTS idx_stage_two_run_watermark
    ON stage_two_run (covered_through DESC)
    WHERE status = 'completed';

-- Counting defects newer than that watermark is the other half of the gate, and
-- it runs on every stage-two trigger. This index also serves the existing
-- DISTINCT ON (object_key) reads of the defect and known-good cohorts.
CREATE INDEX IF NOT EXISTS idx_inference_results_status_inferred_at
    ON inference_results (status, inferred_at DESC);
