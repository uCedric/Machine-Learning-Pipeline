"""Spark adapter: stage-one scoring on the workers via ``predict_batch_udf``.

This is where stage-one's ResNet50 feature extraction happens on the Spark
workers. A batch of events becomes a DataFrame of image *references*, and the
scoring is a :func:`pyspark.ml.functions.predict_batch_udf` — Spark's API for
exactly this shape of work: model inference over a column of data, fed to the
model in mini-batches rather than row by row.

What travels, and what does not
-------------------------------
Only metadata rides in the DataFrame: bucket, object key, content type, size.
Each executor **downloads its own images** from object storage (in parallel,
since a mini-batch of small objects is latency-bound), scores them, renders the
heatmaps, and writes the PNGs back to MinIO and the verdicts to Postgres — right
where the pixels already are. Only the scalar verdict returns to the driver.

Reading the images on the driver and shipping the bytes would move every image
across the network twice and make the driver the memory and bandwidth
bottleneck for the whole batch; fetching them on the executor keeps that traffic
inside the cluster and lets it scale with the number of workers.

The model is loaded once per Python worker, not once per batch
--------------------------------------------------------------
``predict_batch_udf`` takes a *factory* (``make_predict_fn``) rather than a
model: Spark calls it on the worker and reuses the returned closure for the
batches that follow. :func:`_make_predict_fn` resolves its model through
:data:`_STATE`, a module-global built on first use, so the ONNX session and the
FAISS index are constructed **exactly once per Python worker process** — no
matter how often Spark asks for a predict function — and every later micro-batch
reuses them. With ``spark.python.worker.reuse`` (pinned on in the session) those
processes outlive the job, so steady-state batches load nothing at all.

An ``onnxruntime.InferenceSession`` and a FAISS index are unpicklable, which is
the other reason the model is built on the worker instead of shipped: only the
factory function travels, and it travels by reference (the executors import this
module from their own image). Configuration is likewise re-read from the
executor's own environment, so no credential ever rides inside a Spark task.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from adapters.outbound.db import (
    Database,
    PostgresModelRegistry,
    PostgresResultRepository,
)
from adapters.outbound.factory import ModelFactory
from adapters.outbound.minio_storage import MinioObjectStorage
from adapters.outbound.patchcore.heatmap import MatplotlibHeatmapRenderer
from application.ports.batch_scorer import BatchAnomalyScorer, ScoredEvent
from application.ports.storage import ObjectStorage
from application.use_cases.run_inference import RunInferenceUseCase
from config.settings import Settings
from domain.models import InferenceEvent, StageType

logger = logging.getLogger(__name__)

# One DataFrame row per image — a reference, not the pixels. The executor turns
# it back into an InferenceEvent and fetches the object itself.
Row = tuple[str, str, str, int]


@dataclass
class _ExecutorState:
    """Everything one executor Python worker needs, built once and reused."""

    use_case: RunInferenceUseCase
    storage: ObjectStorage
    database: Database
    fetch_workers: int
    device: str


def _task_device(configured: str) -> str:
    """Resolve which device *this* Python worker should run the backbone on.

    On CPU there is nothing to resolve. For ``cuda``, the board is whichever one
    Spark allocated to this task: with ``spark.task.resource.gpu.amount`` set,
    :meth:`TaskContext.resources` reports the addresses reserved for it, and
    pinning to that address is what stops every Python worker on a multi-GPU box
    from piling onto device 0 — each worker is its own process with its own CUDA
    context, so they must not share a board.

    Falls back to the configured string when Spark has no GPU resources
    declared, which keeps a mis-configured cluster running (slowly) instead of
    failing every task.
    """
    if not configured.startswith("cuda") or ":" in configured:
        return configured
    try:
        from pyspark import TaskContext

        context = TaskContext.get()
        addresses = context.resources()["gpu"].addresses if context else []
    except Exception:  # noqa: BLE001 - no GPU resources declared, or no context
        addresses = []
    if not addresses:
        logger.warning(
            "MODEL_DEVICE=%s but Spark reported no GPU for this task; "
            "set spark.executor.resource.gpu.amount and .discoveryScript",
            configured,
        )
        return configured
    return f"cuda:{addresses[0]}"


# Per-Python-worker cache. Spark reuses a worker process across tasks and jobs,
# so this survives between batches and is what makes the model load a one-off.
_STATE: _ExecutorState | None = None


def _executor_state() -> _ExecutorState:
    """Build (once per executor process) the stage-one use case and its ports."""
    global _STATE
    if _STATE is not None:
        return _STATE

    settings = Settings.from_env()
    storage = MinioObjectStorage(settings.minio)
    # A small pool: one executor Python worker scores one batch at a time.
    database = Database(settings.postgres.sql_uri, minconn=1, maxconn=2)

    # Pin the backbone to this worker's own board before building it. The model
    # is cached for the life of the process, so this is decided once: a CUDA
    # context is per process and rebuilding one to chase a reassigned GPU would
    # cost more than running on the original board.
    device = _task_device(settings.model.device)
    registry = PostgresModelRegistry(database)
    model = ModelFactory(registry, storage).build(
        replace(settings.model, device=device)
    )

    use_case = RunInferenceUseCase(
        storage,
        model,
        MatplotlibHeatmapRenderer(),
        PostgresResultRepository(database),
        model_id=model.model_id,
        buffer_zone=model.buffer_zone,
        heatmap_bucket=settings.minio.images_bucket,
        # The use case cannot publish at all: stage-two triggers are emitted by
        # the driver once the batch completes, so an executor never needs broker
        # access. See RunStageOneBatch.
    )
    logger.info(
        "Executor ready: PatchCore model_id=%s on %s (pid=%d)",
        model.model_id,
        device,
        os.getpid(),
    )
    _STATE = _ExecutorState(
        use_case=use_case,
        storage=storage,
        database=database,
        fetch_workers=max(1, settings.spark.fetch_workers),
        device=device,
    )
    return _STATE


def _predict_fn():
    """Return the batch-scoring closure ``predict_batch_udf`` will call.

    Spark invokes this on the worker and caches what it returns, so the model
    load happens off the hot path. The heavy lifting is delegated to
    :func:`_executor_state`, whose module-global cache guarantees a single ONNX
    session and FAISS index per Python worker process even if Spark decides to
    ask for a fresh predict function (e.g. once per task rather than once per
    process) — after the first call this function is a dictionary lookup.
    """
    state = _executor_state()

    def predict(buckets, object_keys, content_types, size_bytes):
        """Score one mini-batch of rows. Runs on the executor.

        Receives one numpy array per input column and returns one entry per
        input row — ``predict_batch_udf`` enforces that alignment, so failures
        are reported in the ``error`` field rather than raised, exactly as
        :class:`~application.ports.batch_scorer.BatchAnomalyScorer` requires.
        """
        events = [
            InferenceEvent(
                bucket=str(bucket),
                object_key=str(object_key),
                content_type=str(content_type),
                size_bytes=int(size),
                event="inference",
                type=StageType.STAGE_ONE,
            )
            for bucket, object_key, content_type, size in zip(
                buckets, object_keys, content_types, size_bytes
            )
        ]
        return _score(state, events)

    return predict


def _fetch(
    state: _ExecutorState, events: Sequence[InferenceEvent]
) -> tuple[dict[int, bytes], list[str | None]]:
    """Download this mini-batch's images on the executor, in parallel.

    A mini-batch is a handful of small objects, so the reads are latency-bound
    and a thread pool collapses them into roughly one round trip. Returns the
    bytes keyed by row index — an unreadable object simply leaves a gap — plus
    the per-row error slots those gaps filled in.
    """
    images: dict[int, bytes] = {}
    errors: list[str | None] = [None] * len(events)
    with ThreadPoolExecutor(max_workers=state.fetch_workers) as pool:
        pending = [
            (index, pool.submit(state.storage.get_object, event.bucket, event.object_key))
            for index, event in enumerate(events)
        ]
        for index, future in pending:
            try:
                images[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                logger.exception(
                    "Failed to read %s/%s",
                    events[index].bucket,
                    events[index].object_key,
                )
                errors[index] = f"{type(exc).__name__}: {exc}"
    return images, errors


def _score(state: _ExecutorState, events: list[InferenceEvent]) -> dict[str, np.ndarray]:
    """Fetch, score and persist one mini-batch. Runs on the executor.

    Three phases, each with its own failure granularity. The download is per
    image, so an unreadable object only loses its own row. The model call is
    batched — one stacked tensor through the backbone, one FAISS search over
    every patch in the batch — and a batch-level failure falls back to scoring
    image by image to isolate the culprit. Persisting (heatmap render, upload,
    result row) is per image again.
    """
    count = len(events)
    statuses: list[str | None] = [None] * count
    scores: list[float] = [0.0] * count

    images, errors = _fetch(state, events)
    # Row indexes of the images that were readable, and their bytes in that
    # order: ``position`` indexes this batch, ``indexes[position]`` the row.
    indexes = sorted(images)
    payloads = [images[index] for index in indexes]

    scored: list = []
    if payloads:
        try:
            scored = state.use_case.score_batch(payloads)
        except Exception:  # noqa: BLE001 - isolate the bad row, keep the good ones
            logger.warning(
                "Batched scoring of %d image(s) failed; retrying image-by-image",
                len(payloads),
            )
            scored = []
            for position, payload in enumerate(payloads):
                try:
                    scored.append(state.use_case.score_batch([payload])[0])
                except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                    index = indexes[position]
                    logger.exception("Failed to score %s", events[index].object_key)
                    errors[index] = f"{type(exc).__name__}: {exc}"
                    scored.append(None)

    for position, outcome in enumerate(scored):
        if outcome is None:
            continue
        index = indexes[position]
        try:
            result = state.use_case.finalise(events[index], *outcome)
            statuses[index] = result.status.value
            scores[index] = result.anomaly_score
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            logger.exception("Failed to store result for %s", events[index].object_key)
            errors[index] = f"{type(exc).__name__}: {exc}"

    return {
        "status": np.array(statuses, dtype=object),
        "anomaly_score": np.array(scores, dtype=np.float32),
        "error": np.array(errors, dtype=object),
    }


class SparkAnomalyScorer(BatchAnomalyScorer):
    """Scores a batch of stage-one events on the cluster with a batch UDF."""

    def __init__(self, spark, udf, *, partitions: int) -> None:
        self._spark = spark
        self._udf = udf
        self._partitions = partitions

    @classmethod
    def initialize(cls, settings: Settings) -> "SparkAnomalyScorer":
        """Open a long-lived Spark session in client mode and build the UDF.

        The driver runs inside this container, so the executors must be able to
        reach it back: ``spark.driver.host`` is the container's name on the
        compose network and the driver/block-manager ports are pinned so they
        can be published.

        The UDF is built once here rather than per batch — it is a description
        of the work (factory, return schema, batch size), so rebuilding it per
        call would only churn closures.
        """
        from pyspark.ml.functions import predict_batch_udf
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName("stage-one")
            .master(settings.spark.master_url)
            .config("spark.driver.host", settings.spark.driver_host)
            .config("spark.driver.bindAddress", "0.0.0.0")
            .config("spark.driver.port", str(settings.spark.driver_port))
            .config("spark.blockManager.port", str(settings.spark.block_manager_port))
            # Keep the Python worker processes alive between tasks. This is what
            # makes :data:`_STATE` worth having: the module-global holds the
            # ONNX session and the FAISS index in that process's memory, but
            # only reuse keeps the process itself around long enough to benefit.
            # Without it every task would fork a fresh worker and reload the
            # backbone. ``true`` is Spark's default — pinned explicitly so a
            # cluster-level spark-defaults.conf cannot silently turn the model
            # cache off.
            .config("spark.python.worker.reuse", "true")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

        udf = predict_batch_udf(
            make_predict_fn=_predict_fn,
            return_type=_return_type(),
            batch_size=settings.spark.udf_batch_size,
        )
        logger.info(
            "Spark session ready on %s (driver=%s, partitions=%d, udf_batch_size=%d)",
            settings.spark.master_url,
            settings.spark.driver_host,
            settings.spark.partitions,
            settings.spark.udf_batch_size,
        )
        return cls(spark, udf, partitions=settings.spark.partitions)

    def score(self, events: Sequence[InferenceEvent]) -> list[ScoredEvent]:
        if not events:
            return []

        by_key = {event.object_key: event for event in events}
        rows: list[Row] = [
            (event.bucket, event.object_key, event.content_type, event.size_bytes)
            for event in events
        ]
        # Cap the slice count at the row count: an empty partition would still
        # pay a model load on its executor for no work. The RDD is partitioned
        # up front and handed to createDataFrame, so no shuffle is needed to
        # place the rows.
        slices = max(1, min(self._partitions, len(rows)))
        logger.info(
            "Dispatching %d image(s) to Spark across %d partition(s)", len(rows), slices
        )

        frame = self._spark.createDataFrame(
            self._spark.sparkContext.parallelize(rows, slices), _input_schema()
        )
        outcomes = (
            frame.select(
                "object_key",
                self._udf("bucket", "object_key", "content_type", "size_bytes").alias(
                    "verdict"
                ),
            )
            .select(
                "object_key", "verdict.status", "verdict.anomaly_score", "verdict.error"
            )
            .collect()
        )

        scored = [
            ScoredEvent(
                event=by_key[row["object_key"]],
                status=row["status"],
                anomaly_score=float(row["anomaly_score"] or 0.0),
                error=row["error"],
            )
            for row in outcomes
            if row["object_key"] in by_key
        ]
        broken = sum(1 for result in scored if not result.ok)
        logger.info(
            "Spark batch complete: %d scored, %d failed", len(scored) - broken, broken
        )
        return scored

    def close(self) -> None:
        self._spark.stop()


def _input_schema():
    """The DataFrame the cluster is handed: where each image lives, not its bytes."""
    from pyspark.sql.types import IntegerType, StringType, StructField, StructType

    return StructType(
        [
            StructField("bucket", StringType(), False),
            StructField("object_key", StringType(), False),
            StructField("content_type", StringType(), False),
            StructField("size_bytes", IntegerType(), False),
        ]
    )


def _return_type():
    """What the UDF gives back per row — scalars only.

    The heatmap is rendered and stored where the image was scored, so the
    verdict is all that travels back: returning the preprocessed tensor or the
    PNG would move hundreds of KB per image to the driver for nothing.
    """
    from pyspark.sql.types import FloatType, StringType, StructField, StructType

    return StructType(
        [
            StructField("status", StringType(), True),
            StructField("anomaly_score", FloatType(), True),
            StructField("error", StringType(), True),
        ]
    )
