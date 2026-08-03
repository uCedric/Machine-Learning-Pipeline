"""Typed configuration assembled from environment variables (12-factor).

All environment reads live here so the rest of the codebase stays free of
``os.getenv`` calls and is easy to test. Configuration is read once at the edge
(the composition roots in :mod:`bootstrap`) and injected inward.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Local directory the model assets are downloaded to before loading. The bucket,
# model_type and version are resolved at runtime from the ``inference_model``
# table (see the model registry), and the assets fetched from object storage
# rather than baked into the image.
_DEFAULT_MODEL_CACHE_DIR = "/tmp/model_assets"


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _required(*keys: str) -> str:
    """Return the first non-empty value among ``keys``.

    Credentials have no safe default and must be supplied via the environment
    (loaded from ``.env``); raise if none of ``keys`` is set.
    """
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    raise RuntimeError(f"Missing required credential; set one of: {', '.join(keys)}")


def _bool(key: str, default: str = "false") -> bool:
    return _env(key, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str
    topic: str
    group_id: str
    event_type: str


@dataclass(frozen=True)
class StageTwoConfig:
    """Stage-two (defect clustering) tuning.

    Stage-two shares the single ``inference`` event/topic; ``trigger_statuses``
    are the stage-one verdicts that make the service re-emit that event as a
    ``stage-two`` trigger — ``anomaly`` only. The uncertain ``pending`` band is
    deliberately absent from the whole clustering path: those images go to a human
    in the monitor dashboard, who can send them for retraining. Adding ``pending``
    back here would only emit triggers that the gate cannot count.

    Those triggers are frequent and clustering is whole-set, so it does not run
    on each one — and when it does run, it picks one of two modes by how much has
    accumulated. ``assign_count`` new ``anomaly`` images since the last run of
    *any* mode projects just those images through the current version's frozen fit
    (cheap, and cluster ids stay comparable). ``refit_count`` new images since the
    last **fit** re-clusters the whole set and freezes it as a new model version
    (expensive, and it renumbers the clusters).

    Two counts means two watermarks, both derived from the ``stage_two_run`` table
    rather than counted in memory — so a restart resumes — and kept separate
    because an assign advancing the fit watermark would reset the re-fit count
    every time and the clusters would never be redrawn.

    ``min_batch`` remains the floor below which a whole-set clustering is
    meaningless (UMAP needs points); with a ``refit_count`` in the hundreds it
    only fires when the counted images turn out to be unfetchable.
    ``normal_set_size`` / ``defect_set_size`` bound the known-good baseline and
    the defect set a run pulls from Postgres — keep ``defect_set_size`` at or
    above ``refit_count``, or a fit will not even see all the images that
    triggered it.
    """

    trigger_statuses: frozenset[str]
    min_batch: int
    refit_count: int
    assign_count: int
    normal_set_size: int
    defect_set_size: int


@dataclass(frozen=True)
class SparkConfig:
    """Where and how stage-one scoring runs on the Spark cluster.

    Stage-one's ResNet50 feature extraction is dispatched to the workers. The
    live Kafka path accumulates a micro-batch of up to ``micro_batch_size``
    events (waiting at most ``poll_timeout_ms`` for them) and sends the batch to
    the cluster, because distributing one image at a time costs more in
    scheduling than it saves in compute. ``partitions`` is how many slices a
    batch is cut into — i.e. how many *tasks*, which is not the same as how many
    copies of the model exist: with ``spark.python.worker.reuse`` a task beyond
    the cluster's slot count simply queues and then runs on an already-warm
    Python worker. What bounds the model copies is the slot count itself
    (``SPARK_WORKER_CORES`` per worker container), since one concurrent task
    means one Python worker holding its own ONNX session and FAISS index.

    Stage-one is batch-only and always runs on the cluster: there is no
    in-process path to fall back to, so the service requires a reachable master.

    A Kafka ``inference`` event is the only way an image reaches stage-one. The
    driver runs inside the inference-service container in client mode, so
    ``driver_host`` and the two pinned ports are how executors call back to it.

    Only image references travel to the cluster; each executor downloads its own
    images, and ``fetch_workers`` is how many of those object-storage reads it
    overlaps (read on the *worker*, so the Spark containers need the variable
    too). ``udf_batch_size`` is how many images ``predict_batch_udf`` feeds the
    model per call — the mini-batch the backbone actually sees, read on the
    driver where the UDF is built. Raising it trades executor memory (a stacked
    ``(B, 3, 224, 224)`` tensor plus ``B * patches`` FAISS queries) for fewer
    model calls.
    """

    master_url: str
    micro_batch_size: int
    poll_timeout_ms: int
    partitions: int
    udf_batch_size: int
    fetch_workers: int
    driver_host: str
    driver_port: int
    block_manager_port: int


@dataclass(frozen=True)
class MinioConfig:
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool
    images_bucket: str


@dataclass(frozen=True)
class PostgresConfig:
    sql_uri: str


@dataclass(frozen=True)
class EltConfig:
    input_folder: str
    processed_dir: str
    poll_interval: float
    run_once: bool


@dataclass(frozen=True)
class ModelConfig:
    """How to build one model, including where its forward pass runs.

    ``device`` is ``cpu``, ``cuda``, or ``cuda:<id>`` for a specific board. The
    id matters on the Spark workers: a Python worker is a separate process with
    its own CUDA context, so each one must be pinned to its own GPU rather than
    all of them piling onto device 0. The composition root resolves the id from
    the task's assigned resources and rebuilds this config with
    :func:`dataclasses.replace`, which is why the field lives here rather than
    being read straight from the environment at build time.
    """

    name: str
    model_key: str
    image_size: int
    cache_dir: str
    device: str = "cpu"


@dataclass(frozen=True)
class Settings:
    kafka: KafkaConfig
    minio: MinioConfig
    postgres: PostgresConfig
    elt: EltConfig
    model: ModelConfig
    cluster_model: ModelConfig
    stage_two: StageTwoConfig
    spark: SparkConfig

    @staticmethod
    def from_env() -> "Settings":
        endpoint = _env("MINIO_ENDPOINT", "minio:9000")
        access_key = _required("MINIO_ACCESS_KEY", "MINIO_ROOT_USER")
        secret_key = _required("MINIO_SECRET_KEY", "MINIO_ROOT_PASSWORD")
        secure = _bool("MINIO_SECURE", "false")

        return Settings(
            kafka=KafkaConfig(
                bootstrap_servers=_env("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"),
                # One 'inference' event/topic drives both stages; the event's
                # 'type' (stage-one / stage-two) selects which process runs. The
                # inference service both consumes this topic and re-publishes to
                # it (a stage-two trigger) — so it is a self-loop on one topic.
                topic=_env("KAFKA_TOPIC", "inference"),
                group_id=_env("KAFKA_GROUP_ID", "inference-service"),
                event_type=_env("KAFKA_EVENT_TYPE", "inference"),
            ),
            minio=MinioConfig(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
                images_bucket=_env("IMAGES_BUCKET", "images"),
            ),
            postgres=PostgresConfig(
                sql_uri=_env("POSTGRES_SQL_URI", _postgres_uri()),
            ),
            elt=EltConfig(
                input_folder=_env("ELT_INPUT_FOLDER", "/data"),
                processed_dir=_env("ELT_PROCESSED_DIR", "/data/processed"),
                poll_interval=float(_env("ELT_POLL_INTERVAL", "5")),
                run_once=_bool("ELT_RUN_ONCE", "true"),
            ),
            model=ModelConfig(
                name=_env("MODEL_NAME", "patchcore"),
                model_key=_env("MODEL_KEY", "patchcore"),
                image_size=int(_env("MODEL_IMAGE_SIZE", "224")),
                cache_dir=_env("MODEL_CACHE_DIR", _DEFAULT_MODEL_CACHE_DIR),
                # 'cuda' asks for GPU inference; the concrete board is resolved
                # per Python worker from the task's assigned resources. Requires
                # an image built with onnxruntime-gpu — the CPU wheel silently
                # has no CUDA provider to fall back from.
                device=_env("MODEL_DEVICE", "cpu"),
            ),
            # Stage-two defect-clustering backbone. Its version/validity is
            # resolved from the inference_model table like the anomaly model's;
            # ResNet50 is the deployed (valid) backbone, DINOv2 a not-yet-valid
            # candidate. CLUSTER_MODEL_IMAGE_SIZE defaults to 448, the resolution
            # the ResNet50 clustering experiment was tuned at.
            cluster_model=ModelConfig(
                name=_env("CLUSTER_MODEL_NAME", "resnet50"),
                model_key=_env("CLUSTER_MODEL_KEY", "resnet50"),
                image_size=int(_env("CLUSTER_MODEL_IMAGE_SIZE", "448")),
                cache_dir=_env("MODEL_CACHE_DIR", _DEFAULT_MODEL_CACHE_DIR),
            ),
            # Stage-one batch runner. SPARK_PARTITIONS defaults to 6 (two per
            # worker); each partition loads its own PatchCore, so raising it
            # past the cluster's core count only adds model-load overhead.
            spark=SparkConfig(
                master_url=_env("SPARK_MASTER_URL", "spark://spark-master:7077"),
                micro_batch_size=int(_env("SPARK_MICRO_BATCH_SIZE", "32")),
                poll_timeout_ms=int(_env("SPARK_POLL_TIMEOUT_MS", "2000")),
                partitions=int(_env("SPARK_PARTITIONS", "6")),
                # 8 images per model call: a 32-image micro-batch cut into 6
                # partitions leaves ~5 rows per task, so one call per task.
                udf_batch_size=int(_env("SPARK_UDF_BATCH_SIZE", "8")),
                fetch_workers=int(_env("SPARK_FETCH_WORKERS", "8")),
                driver_host=_env("SPARK_DRIVER_HOST", "inference-service"),
                driver_port=int(_env("SPARK_DRIVER_PORT", "40001")),
                block_manager_port=int(_env("SPARK_BLOCK_MANAGER_PORT", "40002")),
            ),
            stage_two=StageTwoConfig(
                trigger_statuses=frozenset(
                    s.strip()
                    for s in _env("STAGE_TWO_TRIGGER_STATUSES", "anomaly").split(",")
                    if s.strip()
                ),
                min_batch=int(_env("STAGE_TWO_MIN_BATCH", "15")),
                # New anomaly images required to re-fit (redraw the clusters)
                # and to assign (label new images against the existing ones).
                refit_count=int(_env("STAGE_TWO_REFIT_COUNT", "1000")),
                assign_count=int(_env("STAGE_TWO_ASSIGN_COUNT", "500")),
                normal_set_size=int(_env("STAGE_TWO_NORMAL_SET_SIZE", "200")),
                # Raised to 5000 with the accumulation gate: a run must be able
                # to see at least the refit_count images that opened it.
                defect_set_size=int(_env("STAGE_TWO_DEFECT_SET_SIZE", "5000")),
            ),
        )


def _postgres_uri() -> str:
    user = _required("POSTGRES_USER")
    password = _required("POSTGRES_PASSWORD")
    host = _env("POSTGRES_HOST", "postgres")
    port = _env("POSTGRES_PORT", "5432")
    database = _env("POSTGRES_DB", "postgres")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
