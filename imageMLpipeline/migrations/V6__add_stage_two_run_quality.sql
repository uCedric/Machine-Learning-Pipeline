-- Flyway V6: record how *well* each stage-two run clustered, not just that it ran.
--
-- An assign run can only place an image into a cluster that already exists, so a
-- genuinely new defect type comes back as cluster_id = -1. That makes the share of
-- -1 the system's drift signal: when the newly arriving defects stop resembling
-- the space the fit was built in, noise climbs. Recording it per run is what lets
-- the monitor dashboard show the trend and let a human decide whether the
-- clustering algorithm itself needs rethinking — the pipeline cannot fix that on
-- its own, and should not pretend to.
--
-- Three columns, and the distinction between the first two matters:
--
--   defect_count   (already present) — how many NEW anomalies opened the gate.
--                  A fit's gate is 1000 while it may cluster 5000 images, so this
--                  is *not* a denominator.
--   labelled_count — how many images this run actually gave a cluster_id to. This
--                  is the denominator for the noise ratio.
--   noise_count    — how many of those came back -1.
--
-- mean_probability is HDBSCAN's membership strength averaged over the run: a fall
-- means points are landing at the edges of clusters rather than in them, which is
-- the same drift showing up before it turns into outright noise.

ALTER TABLE stage_two_run
    ADD COLUMN IF NOT EXISTS labelled_count   INTEGER,
    ADD COLUMN IF NOT EXISTS noise_count      INTEGER,
    ADD COLUMN IF NOT EXISTS mean_probability DOUBLE PRECISION;

-- The dashboard reads the newest runs, and joins each assign to the fit whose
-- state it used (stage_two_run.state_model_id) to compare noise ratios.
CREATE INDEX IF NOT EXISTS idx_stage_two_run_recent
    ON stage_two_run (started_at DESC);
