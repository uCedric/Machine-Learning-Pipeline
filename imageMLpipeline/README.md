# Image Inference Pipeline (Kafka · MinIO · Postgres · Spark)

An event-driven image-inference stack designed with Hexagonal Architecture. An **ELT** loads images into object
storage and announces them on Kafka; a **Python inference microservice** consumes
those events, runs a model, and records results in a Postgres table.

## Architecture

```
                   ./data/incoming (drop images here)
                            │
                            ▼
   ┌────────────────────────────────────┐
   │  ELT service (bootstrap.elt)        │
   │   1. load image  ───────────────────────────►  MinIO bucket "images"
   │   2. emit "inference" event ─────────┐
   └──────────────────────────────────────┼──────►  Kafka topic "inference"
                                           │                 │
                                           │                 ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  Inference microservice (bootstrap.inference)                       │
   │   1. consume event   2. read image from MinIO                       │
   │   3. model.predict() → "inferencing", returns 1                     │
   │   4. append result row ─────────────────────────────────────────┐  │
   └──────────────────────────────────────────────────────────────────┼─┘
                                                                       ▼
                          PostgreSQL table  inference_results

   Spark master + 3 workers: retained cluster, NOT used by the inference path.
```

## Services & ports

| Service             | Purpose                                  | Host port(s)          |
|---------------------|------------------------------------------|-----------------------|
| `kafka`             | Event bus (KRaft, no ZooKeeper)          | 9092, 29092           |
| `minio`             | S3-compatible object storage             | 9000 (API), 9001 (UI) |
| `minio-init`        | One-shot: creates `images` + `models`    | —                     |
| `postgres`          | Inference results store                   | 5432                  |
| `elt`               | Image ingest → MinIO → Kafka event       | —                     |
| `inference-service` | Kafka consumer → model → Postgres        | —                     |
| `spark-master`      | Retained Spark cluster (idle)            | 7077, 8081 (UI)       |
| `spark-worker-1..3` | Retained Spark workers (idle)            | 8082 / 8083 / 8084    |

MinIO console: http://localhost:9001 (default `minioadmin` / `minioadmin`).

## Quick start

```bash
# 1. Build images and start the stack
docker compose up -d --build

# 2. Drop your own .png/.jpg files into ./data (the elt service picks up
#    ELT_INPUT_FILE, default ./data/007.png, and uploads it)

# 3. Watch the pipeline work
docker compose logs -f elt inference-service
#    elt:               "Loaded image …" / "Published 'inference' event …"
#    inference-service: "Saved result … score=0.42 (OK), heatmap=heatmap/007.png"

# 4. Inspect the results table
docker compose exec postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT * FROM inference_results;"
```

Stop with `docker compose down` (add `-v` to wipe Kafka/MinIO/Postgres volumes).

## How it flows

1. Drop an image into `./data/incoming`.
2. `elt` uploads it to the MinIO `images` bucket, records the image in the
   Postgres `image` table, then publishes an `inference` event (`{event,
   event_id, bucket, object_key, content_type, size_bytes, created_at}`) to the
   `inference` topic, and moves the file to `./data/processed`.
3. `inference-service` consumes the event, reads the image back from MinIO, runs
   the model, and appends a row to the `inference_results` Postgres table. Kafka
   offsets are committed only after the row is saved (at-least-once).

## Project layout

The `services/` codebase follows **Hexagonal Architecture** (Ports & Adapters).
Dependencies point inward only: `adapters → application → domain`. The domain
imports nothing; the application depends only on the domain and the port
interfaces it owns; concrete infrastructure is reached solely in `bootstrap/`.

```
services/
  domain/                       # ── Core: entities, zero external deps ──
    models.py                     ImageObject, InferenceEvent, InferenceResult
  application/                  # ── Application: orchestration + the ports it needs ──
    ports/
      storage.py                  ObjectStorage
      events.py                   EventPublisher, EventConsumer
      repository.py               ResultRepository
      model.py                    AnomalyModel        (the inference-model port)
      heatmap.py                  HeatmapRenderer
    use_cases/
      ingest_image.py             IngestImage          (ELT use case)
      run_inference.py            RunInferenceUseCase   (depends only on ports)
  adapters/                     # ── Infrastructure: ports & adapters ──
    inbound/                       driving adapters (entry loops)
      elt_poller.py                 poll a path → IngestImage
      inference_consumer.py         consume Kafka → RunInferenceUseCase
    outbound/                      driven adapters (implement ports)
      minio_storage.py              ObjectStorage  → MinIO
      kafka_events.py               EventPublisher/EventConsumer → Kafka
      postgres_repository.py        ResultRepository → Postgres
      patchcore/                    PatchCore model adapter
        model.py                      PatchCore (implements AnomalyModel)
        resources.py                  load memory bank + FAISS + ONNX (stage 1)
        factory.py                    ModelFactory / build_patchcore
        heatmap.py                    MatplotlibHeatmapRenderer (stage 3)
        asset/                        memory_bank.npy, resnet_backbone.onnx(.data)
  config/                       # ── Configuration (read once, at the edge) ──
    settings.py                   env-driven Settings (Kafka/MinIO/Postgres/ELT/Model)
    logging.py                    logging setup
  bootstrap/                    # ── Composition roots: the ONLY wiring ──
    elt.py                        python -m bootstrap.elt
    inference.py                  python -m bootstrap.inference
```

The use cases never import an SDK — they speak only to ports. Swapping the model
means adding one `AnomalyModel` adapter and changing one line in
`bootstrap/inference.py`; swapping object storage or the event bus is the same
one-adapter, one-line change. The two deployables (`elt`, `inference-service`)
share one hexagon and differ only in their composition root.

## Configuration

All settings come from environment variables (see `.env`): Kafka topic, MinIO
credentials/buckets, Postgres connection, ELT poll interval, and the model name.
Defaults work out of the box for local use.

## Future optimizations

The pipeline will move towards **image-centric unsupervised learning** with heavy
image preprocessing and no real columnar/tabular workload. Iceberg (a table format
for columnar analytics) is a poor fit for that direction, so the planned changes are:

- **Drop Iceberg** as a result sink and keep **Postgres** as the single store for
  the small structured result metadata. Image bytes stay in **MinIO** (direct
  key access — no table format needed).
- **Training-data loading:** pack the large number of small image files into
  shard/columnar-ML formats — **WebDataset** (tar shards, streaming) or **Lance**
  (ML-native columnar with random access) — instead of reading millions of
  individual objects.
- **Dataset versioning / reproducibility:** version the images and preprocessing
  outputs on object storage with **lakeFS** or **DVC** so each training run is
  reproducible. (Iceberg only versions tables, not image blobs.)
