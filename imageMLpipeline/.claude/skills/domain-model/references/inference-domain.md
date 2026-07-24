# Inference service — domain entities

Source: `inference-service/domain/models.py`. These are the business objects of
the *prediction* side: an image arrives, the model scores it, the buffer zone
turns the score into a verdict, and a result is recorded. All are immutable
(`@dataclass(frozen=True)`) and framework-free.

## PredictionStatus — the verdict

A closed set of three outcomes: `normal`, `pending`, `anomaly`. Implemented as a
`StrEnum`, so the member *is* its lowercase string — it serialises cleanly and
maps one-to-one onto the `prediction_status` Postgres enum and the
`inference_results.status` column. No other verdict exists.

## BufferZone — the decision rule

A value object holding two floats, `lower` and `upper`, with one behaviour,
`classify(anomaly_score) -> PredictionStatus`:

- `score < lower` → `NORMAL` (clearly good)
- `score > upper` → `ANOMALY` (clearly defective)
- otherwise (`lower <= score <= upper`) → `PENDING` (uncertain; human review)

The pending band is inclusive of both bounds. This is the system's central
business rule — a hysteresis band rather than a single cut-point, so borderline
images are flagged for a person instead of being force-classified. The bounds
are produced/recalibrated by the train service (P75/P99 of a rolling validation
set of known-good images); this service only *applies* them.

## ModelVersion — which model produced a verdict

Identifies a registered model version (a row in the `inference_model` table) and
locates its assets in object storage.

- `model_id` — the row's identity.
- `model_type` — e.g. `patchcore`; **also the object-key prefix** for assets.
- `bucket` — the object-storage bucket holding the assets.
- `version` — the semver **string** (e.g. `1.1.0`).

Assets live at `{bucket}/{model_type}/{version}/`. Note: inference holds the
version as an opaque string because it only needs to *find* the assets — it never
does version arithmetic (contrast the train service's split into integers).

## ImageObject — an image in storage

`bucket`, `key`, `content_type`, `size_bytes`. A plain locator/descriptor for an
image held in object storage.

## InferenceEvent — "this image is ready to be inferred"

Emitted by the ELT once an image has landed in object storage; consumed by the
inference service to trigger a prediction.

- `bucket`, `object_key`, `content_type`, `size_bytes` — where/what the image is.
- `event` — the event-type tag.
- `created_at` — defaults to "now" (UTC).

Business rule: the image is identified solely by its `object_key`, which is a
**uuid-based file name**. Because that key is already unique, the event carries
**no separate event id**.

## InferenceResult — the outcome of one prediction

One instance per inference execution (one row in `inference_results`).

- `event_id` — the result's own identity (defaults to a fresh UUID). One per run.
- `object_key` — the source image (uuid-based file name); `bucket` — its bucket.
- `anomaly_score` — the raw score the model produced.
- `status` — the `PredictionStatus` the buffer zone assigned to that score.
- `model_id` — which `ModelVersion` produced it.
- `heatmap_key` — object key of the rendered explanation heatmap.
- `inferred_at` — defaults to "now" (UTC).

Business rule: a result always pairs the raw `anomaly_score` with the `status`
derived from it, so a recorded verdict is always traceable back to its score and
the model that produced it.
