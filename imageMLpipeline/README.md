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
   the model, and appends a row to the `inference_results` Postgres table. When
   the DINOv2 clustering stage is active (see below) it then also assigns the
   image to a defect cluster and appends a row to `cluster_results`. Kafka
   offsets are committed only after the row is saved (at-least-once).

## Second model: DINOv2 defect clustering

Alongside PatchCore (anomaly detection), the inference service can run a second
model on every consumed image: a **DINOv2 ViT-S/14** embedder followed by a
pretrained **PCA → UMAP → HDBSCAN** pipeline that assigns the image to a defect
cluster.

```
image ─► DINOv2 (ONNX) CLS embedding (384-d) ─► L2 normalise
      ─► PCA 384→50 ─► UMAP 50→2 ─► HDBSCAN approximate_predict
      ─► cluster_results row (cluster_id, probability)
```

- Cluster results land in their own Postgres table, `cluster_results` (one row
  per clustering execution, modelled on `inference_results`; join the two
  streams on `image_id`).
- `cluster_id = -1` means the image matched no known cluster (HDBSCAN noise).
  **No row** for an image means the clustering stage did not run.
- A clustering failure never fails the primary anomaly result: the error is
  logged and only the cluster row is skipped.

The stage is **disabled by default**: at start-up the service resolves the
newest valid `dinov2` version from the `inference_model` table; while none
exists it logs `defect clustering disabled` and behaves exactly as before.

### Activating the clustering stage

1. Export the backbone (dev machine, needs internet):
   `python inference-service/scripts/export_dinov2_onnx.py --output dinov2_vits14.onnx`
2. Fit the clustering artifacts offline — `pca.joblib`, `umap.joblib`,
   `hdbscan.joblib`. Hard requirements:
   - fit HDBSCAN with `prediction_data=True` (the service refuses the artifact
     otherwise);
   - extract the training features with the same preprocessing the service
     uses (shorter-side resize 224 → centre crop 224 → ImageNet normalise →
     L2 norm), ideally with the exported ONNX model itself;
   - use the exact library versions pinned in
     `inference-service/requirements.txt` — the joblib pickles are
     version-coupled. Safest is to fit inside the service image, e.g.
     `docker compose run --rm -v "$PWD:/work" inference-service python /work/fit_clusters.py`.
3. Upload the four files to the MinIO `models` bucket under `dinov2/1.0.0/`
   (console at http://localhost:9001, or `mc cp`).
4. Register the version — this is the activation gate:
   ```sql
   INSERT INTO inference_model
       (model_id, model_type, bucket, major_version, minor_version, patch_version, is_valid)
   VALUES (gen_random_uuid(), 'dinov2', 'models', 1, 0, 0, true);
   ```
5. `docker compose restart inference-service` — the logs show the dinov2 asset
   fetch, a one-off UMAP/HDBSCAN warm-up, then `Saved cluster result …` per
   image.

## Project layout

The `inference-service/` codebase follows **Hexagonal Architecture** (Ports & Adapters).
Dependencies point inward only: `adapters → application → domain`. The domain
imports nothing; the application depends only on the domain and the port
interfaces it owns; concrete infrastructure is reached solely in `bootstrap/`.

```
inference-service/
  domain/                       # ── Core: entities, zero external deps ──
    models.py                     ImageObject, InferenceEvent, InferenceResult,
                                  ClusterAssignment, ClusterResult
  application/                  # ── Application: orchestration + the ports it needs ──
    ports/
      storage.py                  ObjectStorage
      events.py                   EventPublisher, EventConsumer
      repository.py               ResultRepository
      cluster_repository.py       ClusterResultRepository
      model.py                    AnomalyModel        (the anomaly-model port)
      clustering.py               ClusterModel        (the defect-clustering port)
      heatmap.py                  HeatmapRenderer
    use_cases/
      ingest_image.py             IngestImage          (ELT use case)
      run_inference.py            RunInferenceUseCase   (depends only on ports)
  adapters/                     # ── Infrastructure: ports & adapters ──
    inbound/                       driving adapters (entry loops)
      elt_poller.py                 poll a path → IngestImage
      inference_consumer.py         consume Kafka → RunInferenceUseCase
    outbound/                      driven adapters (implement ports)
      factory.py                    ModelFactory (one builder per model type)
      minio_storage.py              ObjectStorage  → MinIO
      kafka_events.py               EventPublisher/EventConsumer → Kafka
      postgres_repository.py        ResultRepository → Postgres
      postgres_cluster_repository.py  ClusterResultRepository → Postgres
      patchcore/                    PatchCore anomaly-model adapter
        model.py                      PatchCore (implements AnomalyModel)
        resources.py                  load memory bank + FAISS + ONNX (stage 1)
        builder.py                    build_patchcore (registered in the factory)
        heatmap.py                    MatplotlibHeatmapRenderer (stage 3)
        asset/                        memory_bank.npy, resnet_backbone.onnx(.data)
      dinov2/                       DINOv2 defect-clustering adapter
        model.py                      Dinov2ClusterModel (implements ClusterModel)
        resources.py                  load ONNX + PCA/UMAP/HDBSCAN artifacts
        builder.py                    build_dinov2 (registered in the factory)
  config/                       # ── Configuration (read once, at the edge) ──
    settings.py                   env-driven Settings (Kafka/MinIO/Postgres/ELT/Model)
    logging.py                    logging setup
  bootstrap/                    # ── Composition roots: the ONLY wiring ──
    elt.py                        python -m bootstrap.elt
    inference.py                  python -m bootstrap.inference
  scripts/                      # ── Dev-only utilities (never in the image) ──
    export_dinov2_onnx.py         export DINOv2 ViT-S/14 to ONNX (run offline)
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
