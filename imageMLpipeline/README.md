# Image Inference Pipeline (Kafka · MinIO · Iceberg · Spark)

An event-driven image-inference stack designed with Hexagonal Architecture. An **ELT** loads images into object
storage and announces them on Kafka; a **Python inference microservice** consumes
those events, runs a model, and records results in an Apache Iceberg table.

## Architecture

```
                   ./data/incoming (drop images here)
                            │
                            ▼
   ┌────────────────────────────────────┐
   │  ELT service (bootstrap.elt)        │
   │   1. load image  ───────────────────────────►  MinIO bucket "images"
   │   2. emit "inference" event ─────────┐
   └──────────────────────────────────────┼──────►  Kafka topic "inference-events"
                                           │                 │
                                           │                 ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  Inference microservice (bootstrap.inference)                       │
   │   1. consume event   2. read image from MinIO                       │
   │   3. model.predict() → "inferencing", returns 1                     │
   │   4. append result row ─────────────────────────────────────────┐  │
   └──────────────────────────────────────────────────────────────────┼─┘
                                                                       ▼
                          Apache Iceberg table  inference.results
                          ├─ data + metadata  →  MinIO bucket "warehouse"
                          └─ catalog          →  PostgreSQL

   Spark master + 3 workers: retained cluster, NOT used by the inference path.
```

## Services & ports

| Service             | Purpose                                  | Host port(s)          |
|---------------------|------------------------------------------|-----------------------|
| `kafka`             | Event bus (KRaft, no ZooKeeper)          | 9092, 29092           |
| `minio`             | S3-compatible object storage             | 9000 (API), 9001 (UI) |
| `minio-init`        | One-shot: creates `images` + `warehouse` | —                     |
| `postgres`          | Iceberg catalog backend                  | 5432                  |
| `elt`               | Image ingest → MinIO → Kafka event       | —                     |
| `inference-service` | Kafka consumer → model → Iceberg         | —                     |
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

# 4. Inspect the Iceberg results table
docker compose run --rm inference-service python -m bootstrap.show_results
```

Stop with `docker compose down` (add `-v` to wipe Kafka/MinIO/Postgres volumes).

## How it flows

1. Drop an image into `./data/incoming`.
2. `elt` uploads it to the MinIO `images` bucket and publishes an `inference`
   event (`{event, event_id, bucket, object_key, content_type, size_bytes,
   created_at}`) to the `inference-events` topic, then moves the file to
   `./data/processed`.
3. `inference-service` consumes the event, reads the image back from MinIO, runs
   the model, and appends a row to the `inference.results` Iceberg table. Kafka
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
      iceberg_repository.py         ResultRepository → Iceberg
      composite_repository.py       ResultRepository → fan-out (Iceberg + Postgres)
      patchcore/                    PatchCore model adapter
        model.py                      PatchCore (implements AnomalyModel)
        resources.py                  load memory bank + FAISS + ONNX (stage 1)
        factory.py                    ModelFactory / build_patchcore
        heatmap.py                    MatplotlibHeatmapRenderer (stage 3)
        asset/                        memory_bank.npy, resnet_backbone.onnx(.data)
  config/                       # ── Configuration (read once, at the edge) ──
    settings.py                   env-driven Settings (Kafka/MinIO/Iceberg/ELT/Model)
    logging.py                    logging setup
  bootstrap/                    # ── Composition roots: the ONLY wiring ──
    elt.py                        python -m bootstrap.elt
    inference.py                  python -m bootstrap.inference
    show_results.py               dev helper: print the results table
```

The use cases never import an SDK — they speak only to ports. Swapping the model
means adding one `AnomalyModel` adapter and changing one line in
`bootstrap/inference.py`; swapping object storage or the event bus is the same
one-adapter, one-line change. The two deployables (`elt`, `inference-service`)
share one hexagon and differ only in their composition root.

## Configuration

All settings come from environment variables (see `.env`): Kafka topic, MinIO
credentials/buckets, Postgres catalog connection, Iceberg namespace/table, ELT
poll interval, and the model name. Defaults work out of the box for local use.
