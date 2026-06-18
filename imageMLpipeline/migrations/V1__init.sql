-- Flyway V1: baseline schema for the inference store.
-- Applied by the `flyway` service; the Postgres adapter assumes these exist.

-- lookup table for inference_model --
CREATE TABLE IF NOT EXISTS model_type (
    code TEXT PRIMARY KEY
);

INSERT INTO model_type (code)
VALUES ('patchcore')
ON CONFLICT (code) DO NOTHING;

-- image schema --
CREATE TABLE IF NOT EXISTS image (
    image_id      UUID PRIMARY KEY,
    object_key    TEXT NOT NULL,
    bucket        TEXT NOT NULL,
    content_type  TEXT NOT NULL,
    size_bytes    BIGINT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- inference_model schema --
CREATE TABLE IF NOT EXISTS inference_model (
    model_id      UUID PRIMARY KEY,
    model_type    TEXT NOT NULL REFERENCES model_type(code),
    bucket        TEXT NOT NULL,
    major_version INTEGER NOT NULL,
    minor_version INTEGER NOT NULL,
    patch_version INTEGER NOT NULL,
    is_valid      BOOLEAN NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- initialization of inference_model table --
INSERT INTO inference_model
    (model_id, model_type, bucket, major_version, minor_version, patch_version, is_valid)
VALUES
    ('2526d2bd-0725-43a2-a29b-4905fb0e5335', 'patchcore', 'models', 1, 0, 0, true)
ON CONFLICT (model_id) DO NOTHING;

-- prediction verdict: a closed, fixed set of outcomes from the buffer zone --
CREATE TYPE prediction_status AS ENUM ('normal', 'pending', 'anomaly');

-- inference_results schema --
CREATE TABLE IF NOT EXISTS inference_results (
    event_id      UUID PRIMARY KEY,
    image_id      UUID NOT NULL REFERENCES image(image_id),
    model_id      UUID NOT NULL REFERENCES inference_model(model_id),
    object_key    TEXT NOT NULL,
    bucket        TEXT NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    status        prediction_status NOT NULL,
    heatmap_key   TEXT NOT NULL,
    inferred_at   TIMESTAMPTZ NOT NULL
);