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

**`PENDING` means "a human decides", and nothing automated consumes it.** It is
excluded from the entire stage-two path: it does not emit a stage-two trigger, is
not counted by the accumulation gate, and is not part of the defect set that gets
clustered. Its route is the monitor dashboard, where an operator reviews the image
and can publish a `train` event to retrain the memory bank. Only `ANOMALY` — a
*confirmed* defect — feeds defect-type clustering, so unconfirmed images cannot
blur the defect types stage-two is trying to name.

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

The single event value object, reused across both stages via its `event` tag:

- The ELT emits it tagged **`stage-one-inference`** once an image lands in object
  storage; stage-one (PatchCore) consumes it to trigger a prediction.
- Stage-one re-emits it tagged **`stage-two-inference`** (same image fields, new
  tag) for every verdict in the trigger set (**`anomaly` only**); stage-two
  consumes it as a trigger, and acts on it only once enough confirmed defects
  have accumulated.

Fields:

- `bucket`, `object_key`, `content_type`, `size_bytes` — where/what the image is.
- `event` — the event-type tag (`stage-one-inference` / `stage-two-inference`).
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

## Stage-two clustering — ClusterAssignment & ClusterResult

Stage-two groups the *confirmed* defective images (stage-one verdict `anomaly`)
into defect types. It is **whole-set batch** clustering — a `stage-two-inference`
event only triggers a run; the service re-reads the full defect set and a
known-good baseline and clusters them together (no per-image predict).

- **ClusterAssignment** — the raw verdict a clustering model produces for one
  image: `cluster_id`, `probability` (membership strength in `[0, 1]`), `model_id`.
  `cluster_id = -1` follows HDBSCAN semantics — noise, i.e. the image matched no
  cluster.
- **ClusterResult** — one row in `cluster_results`, mirroring `InferenceResult`:
  `event_id` (its own identity), `object_key` + `bucket` (the source image),
  `cluster_id`, `probability`, `model_id`, and `clustered_at`. All images in one
  batch run share the same `clustered_at`, so the latest clustering for a
  `model_id` is the rows with the newest `clustered_at`.

Business rules: **no row** for an image means clustering has not run for it; a
`cluster_id` of `-1` is a real verdict (ran, matched no cluster). Because each run
re-clusters the whole set, cluster ids are only meaningful **within** a run, not
across runs. Which backbone produced a row is carried by `model_id` — ResNet50 is
the deployed backbone; DINOv2 is a candidate whose `inference_model` row stays
`is_valid = false` until an evaluate-service promotes it.
