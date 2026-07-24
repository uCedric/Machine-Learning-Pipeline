-- Flyway V2: DINOv2 defect-clustering support.
-- Registers the 'dinov2' model type and creates the cluster_results table
-- (modelled on inference_results: one row per clustering execution).
-- No row for an image means clustering did not run for it; cluster_id = -1 is
-- a real verdict (HDBSCAN noise — the image matched no known cluster).
-- NOTE: no inference_model row is seeded for 'dinov2' — inserting a valid row
-- once the assets are uploaded to MODELS_BUCKET/dinov2/{version}/ is the
-- activation gate for the clustering stage.

INSERT INTO model_type (code)
VALUES ('dinov2')
ON CONFLICT (code) DO NOTHING;

-- cluster_results schema --
CREATE TABLE IF NOT EXISTS cluster_results (
    event_id      UUID PRIMARY KEY,
    image_id      UUID NOT NULL REFERENCES image(image_id),
    model_id      UUID NOT NULL REFERENCES inference_model(model_id),
    object_key    TEXT NOT NULL,
    bucket        TEXT NOT NULL,
    cluster_id    INTEGER NOT NULL,
    probability   DOUBLE PRECISION NOT NULL,
    clustered_at  TIMESTAMPTZ NOT NULL
);
