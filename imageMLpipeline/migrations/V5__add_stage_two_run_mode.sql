-- Flyway V5: stage-two gains two modes — fit and assign.
--
-- Until now every stage-two run re-fitted the whole pipeline (prototype,
-- scalers, PCA, UMAP, HDBSCAN) from scratch, which made cluster ids incomparable
-- between runs. A run now either:
--
--   * fit    — cluster the whole defect set, then freeze the fitted state as a
--              new inference_model version (artifacts in MinIO). Expensive.
--   * assign — project only the newly arrived defects through the frozen state
--              and give them ids from the existing clusters. Cheap, and the ids
--              stay comparable with every earlier assign against that version.
--
-- Two thresholds, and therefore two watermarks, both still derived from this
-- table rather than counted in memory:
--
--   assign gate : new anomalies since the last completed run of *either* mode
--   fit gate    : new anomalies since the last completed run of mode = 'fit'
--
-- Keeping them separate is what makes the pair work: if an assign advanced the
-- fit watermark too, the count would reset every 500 images and never reach the
-- re-fit threshold.

CREATE TYPE stage_two_run_mode AS ENUM ('fit', 'assign');

-- Existing rows (if any) were whole-set re-fits by definition.
ALTER TABLE stage_two_run
    ADD COLUMN IF NOT EXISTS mode stage_two_run_mode NOT NULL DEFAULT 'fit';

-- The version whose frozen state the run used ('assign') or produced ('fit').
-- Nullable because a failed fit never gets as far as registering one.
ALTER TABLE stage_two_run
    ADD COLUMN IF NOT EXISTS state_model_id UUID REFERENCES inference_model(model_id);

-- The fit gate reads the newest completed *fit* watermark; the assign gate reads
-- the newest completed watermark of any mode (served by idx_stage_two_run_watermark
-- from V4).
CREATE INDEX IF NOT EXISTS idx_stage_two_run_fit_watermark
    ON stage_two_run (covered_through DESC)
    WHERE status = 'completed' AND mode = 'fit';
