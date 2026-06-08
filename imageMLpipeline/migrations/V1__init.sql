-- Flyway V1: baseline schema for the inference store.
-- Applied by the `flyway` service; the Postgres adapter assumes these exist.

CREATE TABLE IF NOT EXISTS inference_results (
    event_id      UUID PRIMARY KEY,
    object_key    TEXT NOT NULL,
    bucket        TEXT NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    is_anomaly    BOOLEAN NOT NULL,
    model_name    TEXT NOT NULL,
    heatmap_key   TEXT NOT NULL,
    inferred_at   TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS inference_model (
    model_id      UUID PRIMARY KEY,
    bucket        CHAR(64) NOT NULL,
    major_version INTEGER NOT NULL,
    minor_version INTEGER NOT NULL,
    patch_version INTEGER NOT NULL,
    is_valid      BOOLEAN NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
