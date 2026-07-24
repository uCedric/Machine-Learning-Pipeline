# Train service — domain entities

Source: `train-service/domain/models.py`. These are the business objects of the
*learning* side: a retrain is requested for a model on a new image, and the
outcome is a new candidate model version. All are immutable
(`@dataclass(frozen=True)`) and framework-free.

## TrainEvent — "retrain this model on this image"

Signals that a (re)training run has been requested. It carries the raw message
`payload` as-is and exposes the well-known fields through helpers.

Payload shape:

```json
{"event": "train", "model": "patchcore", "images": "/{bucket}/{key}"}
```

- `event` — the event-type tag (`"train"`).
- `payload` — the raw dict.

Helpers (the business rules for reading the payload):

- `model_type` → `payload["model"]`. The model type to retrain; doubles as the
  object-key prefix in the models bucket (same convention as `ModelVersion`).
- `image_location()` → parses `payload["images"]`, a path of the form
  `/{bucket}/{key}`, into a `(bucket, key)` tuple. The leading `/` root is
  dropped and everything after the first segment is rejoined as the key (so keys
  may contain `/`). **Rule:** a path with fewer than two segments is malformed
  and raises `ValueError` — there is no silent fallback.

## ModelVersion — a registered version, built for version arithmetic

Identifies a registered model version (a row in `inference_model`) and locates
its assets, like the inference-side `ModelVersion`, but models the version
**numerically** so a retrain can compute the next one.

- `model_id` — the row's identity.
- `model_type` — e.g. `patchcore`; also the object-key prefix.
- `bucket` — the assets bucket.
- `major`, `minor`, `patch` — the semver split into integers.
- `version` (property) → `"{major}.{minor}.{patch}"`, e.g. `1.0.0`.

Business rule: the semver is split because a retrain derives the **next patch**
version off a base (e.g. `1.0.0` → `1.0.1`). Assets live at
`{bucket}/{model_type}/{version}/`. Contrast inference, which keeps `version` as
an opaque string because it never increments it.

## ImageObject — an image in storage

`bucket`, `key`, `content_type`, `size_bytes`. A plain locator/descriptor for an
image held in object storage (mirrors the inference-side `ImageObject`).

## How these flow through a retrain (business outcome)

A `TrainEvent` names a `model_type` and an image. The service resolves the
newest valid `ModelVersion` (the base), extends its memory bank with the new
image's features, recalibrates the buffer-zone bounds from a rolling validation
set of known-good images, and writes the result as the **next patch**
`ModelVersion` registered as *not yet valid* — a candidate awaiting promotion.
The domain objects here define that vocabulary; the use case orchestrates it.
