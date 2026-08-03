---
name: domain-model
description: Business vocabulary and rules of the imageMLpipeline domain layer — the anomaly-score buffer zone, prediction verdicts, model versions, and the events/results that flow between the inference and train services. This skill should be used when reasoning about what the system decides and why (not how it is wired), when adding or changing a domain entity, or when a change touches the meaning of a score, a verdict, a version, or an event payload.
---

This skill encapsulates the *business logic* held in the services' `domain/`
folders (`inference-service/domain/models.py`, `train-service/domain/models.py`)
as human-readable rules. The domain layer is the centre of the hexagon: plain,
immutable data structures (frozen dataclasses) with **no** dependency on any
framework, adapter, port, or configuration. It is the vocabulary the rest of the
codebase speaks. When the rules below and the code disagree, the code wins —
update this skill.

## The one decision the system exists to make

Given an image, decide whether the manufactured item is **good**, **defective**,
or **uncertain enough to need a human**. Everything else (storage, Kafka, ONNX,
FAISS, Postgres) is plumbing around that decision.

The decision is made by an **anomaly score** (how far the image is from "known
good") passed through a **buffer zone**. This is **stage-one** (triggered by the
`stage-one-inference` event). A **defective/uncertain** verdict then triggers
**stage-two** (`stage-two-inference`): a whole-set batch clustering that groups the
defective images into defect *types*. Stage-one asks "is this defective?";
stage-two asks "what kind of defect?". See the inference-domain reference for the
stage-two entities (`ClusterAssignment`, `ClusterResult`).

### Buffer zone — the core rule

A `BufferZone(lower, upper)` is a two-bound decision band, not a single
threshold. It deliberately splits the score line into three regions:

| Condition | Verdict | Meaning |
|-----------|---------|---------|
| `score < lower` | `NORMAL` | Clearly good — below the good-product defence line |
| `score > upper` | `ANOMALY` | Clearly defective |
| `lower <= score <= upper` | `PENDING` | Uncertain — route to a human |

This hysteresis design avoids a knife-edge threshold that would flip-flop near
the boundary, and it surfaces ambiguous cases instead of silently guessing.

**Where the bounds come from:** they are *not* hand-tuned constants. The
train-service recalibrates them on every retrain from a **rolling validation set**
of recent known-good images: it scores those images against the freshly updated
memory bank and takes `lower = P75`, `upper = P99` of that good-image score
distribution. So `lower` is "where the looser 25% of genuinely-good images sit"
(a tolerant good-product line) and `upper` is "the near-worst good image" (above
it, almost certainly a real defect). See the `train-domain` reference and the
training use case for the recompute path.

### Prediction verdict — a closed set

`PredictionStatus` is a closed, fixed set of three outcomes: `normal`,
`pending`, `anomaly`. It mirrors the `prediction_status` Postgres enum exactly
and is a `StrEnum`, so each member *is* its lowercase string. No other verdict
is valid anywhere in the system.

## Shared concepts across both services

- **ModelVersion** — a registered model in the `inference_model` table. Its
  assets live in object storage at `{bucket}/{model_type}/{version}/`, where
  `model_type` (e.g. `patchcore`) doubles as the object-key prefix. The two
  services model the *version* differently — inference carries it as a semver
  **string**, train splits it into `major/minor/patch` integers so a retrain can
  derive the next patch number. See the per-service references.
- **ImageObject** — an image located in object storage by `bucket` + `key`,
  with its `content_type` and `size_bytes`.
- **Events carry intent, results carry outcomes.** An event ("this image is
  ready", "retrain this model") enters a service; a result/new-version is what
  leaves. Events and results are immutable value objects.

## References

Read the per-service file before changing that service's domain:

- [references/inference-domain.md](references/inference-domain.md) — the
  inference service's entities: `PredictionStatus`, `BufferZone`, `ModelVersion`
  (string version), `InferenceEvent`, `InferenceResult`, `ImageObject`.
- [references/train-domain.md](references/train-domain.md) — the train service's
  entities: `TrainEvent` (payload shape + `image_location()` parsing rule),
  `ModelVersion` (semver split + `version` property), `ImageObject`.

## Principles for changing the domain

**The domain imports nothing inward-facing.** No adapter, port, or config import
may appear in `domain/`. If a change needs one, it belongs in a port/adapter, not
here.

**Entities are immutable.** All are `@dataclass(frozen=True)`. Derive new values;
never mutate.

**Keep the two ModelVersions intentionally different.** Inference reads a version
string; train does version *arithmetic*. Do not unify them without a reason — the
split mirrors their jobs.

**Verdicts and statuses are contracts with the database.** `PredictionStatus`
values must stay in lockstep with the `prediction_status` SQL enum and the
`inference_results.status` column. Changing one means a migration for the other.

**A score only has meaning through a buffer zone.** Never compare a raw
`anomaly_score` to an ad-hoc number; classify it with `BufferZone.classify`.
