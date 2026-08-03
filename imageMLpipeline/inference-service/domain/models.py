"""Domain entities and value objects.

These are plain, immutable data structures with no dependency on any framework
or external SDK. They are the vocabulary the rest of the application speaks and
sit at the centre of the hexagon — nothing here imports an adapter, a port or
configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PredictionStatus(StrEnum):
    """The verdict for an inferred image.

    ``StrEnum`` so the value *is* its lowercase string (``"normal"`` etc.),
    matching the ``prediction_status`` Postgres enum and serialising cleanly.
    """

    NORMAL = "normal"
    PENDING = "pending"
    ANOMALY = "anomaly"


class StageType(StrEnum):
    """Which inference process an ``inference`` event should activate.

    The stages share one Kafka event; ``type`` is the discriminator the single
    inference service dispatches on. ``StrEnum`` so the value *is* its string
    (``"stage-one"`` / ``"stage-two"`` / ``"refresh"``) and serialises cleanly.
    """

    STAGE_ONE = "stage-one"
    STAGE_TWO = "stage-two"
    REFRESH = "refresh"  # triggers a model refresh (rebuild clustering backbone)


@dataclass(frozen=True)
class BufferZone:
    """A two-bound decision band over the anomaly score.

    Replaces a single threshold with a hysteresis-style buffer: scores below
    ``lower`` are clearly normal, scores above ``upper`` are clearly anomalous,
    and scores inside ``[lower, upper]`` are uncertain (``PENDING``) and worth a
    human look. The bounds are inclusive of the pending band.
    """

    lower: float
    upper: float

    def classify(self, anomaly_score: float) -> PredictionStatus:
        if anomaly_score < self.lower:
            return PredictionStatus.NORMAL
        if anomaly_score > self.upper:
            return PredictionStatus.ANOMALY
        return PredictionStatus.PENDING


@dataclass(frozen=True)
class ImageObject:
    """An image that lives in object storage."""

    bucket: str
    key: str
    content_type: str
    size_bytes: int


@dataclass(frozen=True)
class ModelVersion:
    """A registered model version, as recorded in the ``inference_model`` table.

    Locates the model's assets in object storage: they live in ``bucket`` under
    ``{model_type}/{version}/``. ``model_type`` doubles as the object-key prefix
    (e.g. ``patchcore``) and ``version`` is the semver string (e.g. ``1.1.0``).
    """

    model_id: str
    model_type: str
    bucket: str
    version: str


@dataclass(frozen=True)
class InferenceEvent:
    """Signals that a stored image is ready to be inferred.

    A single event drives both inference stages; ``type`` (a :class:`StageType`)
    selects which one. The ELT emits it as ``stage-one``; stage-one re-emits the
    same event as ``stage-two`` for the images that need clustering. The image is
    identified by its ``object_key`` — a uuid-based file name — so no separate
    event id is carried.
    """

    bucket: str
    object_key: str
    content_type: str
    size_bytes: int
    event: str
    type: str = StageType.STAGE_ONE
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class InferenceResult:
    """The outcome of running the model against a single image.

    Identified by its own ``event_id`` (one row per inference execution). The
    source image is ``object_key`` (a uuid-based file name) and the model is
    referenced by ``model_id``.
    """

    bucket: str
    object_key: str
    anomaly_score: float
    status: PredictionStatus
    model_id: str
    heatmap_key: str
    event_id: str = field(default_factory=lambda: str(uuid4()))
    inferred_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class ClusterAssignment:
    """The defect-cluster verdict a clustering model produced for one image.

    ``cluster_id`` follows HDBSCAN semantics: ``-1`` means noise — the image
    matched no known cluster. ``probability`` is the membership strength in
    ``[0, 1]``, and ``model_id`` references the clustering model version that
    produced the assignment.
    """

    cluster_id: int
    probability: float
    model_id: str


@dataclass(frozen=True)
class ClusterResult:
    """The outcome of clustering a single image (one row in ``cluster_results``).

    Mirrors :class:`InferenceResult`: identified by its own ``event_id`` (one
    row per clustering execution), locates the source image by ``object_key``
    (a uuid-based file name) and ``bucket``, and references the clustering
    model by ``model_id``. No row for an image means clustering did not run;
    ``cluster_id`` ``-1`` means it ran and the image matched no cluster.
    """

    bucket: str
    object_key: str
    cluster_id: int
    probability: float
    model_id: str
    event_id: str = field(default_factory=lambda: str(uuid4()))
    clustered_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class ClusteringRun:
    """One stage-two execution (one row in ``stage_two_run``).

    Stage-two does not cluster on every trigger — it waits for defect images to
    accumulate. This is the record that makes that gate work: ``covered_through``
    is the newest ``inferred_at`` the run took into account, so the next gate
    counts only defects newer than it.

    A ``failed`` run is recorded too, but its watermark is ignored by the gate:
    nothing was persisted, so that backlog must stay pending and be retried.
    ``cluster_count`` is ``None`` for a failed run, and excludes ``-1`` noise
    for a completed one.
    """

    model_id: str
    covered_through: datetime
    defect_count: int
    # 'fit' re-fitted the whole pipeline and froze it as a new model version;
    # 'assign' projected only the new defects through an existing frozen state.
    mode: str = "fit"
    # The version whose frozen state the run produced (fit) or used (assign).
    state_model_id: str | None = None
    status: str = "completed"
    cluster_count: int | None = None
    # How many images this run actually labelled, and how many of those matched no
    # cluster. Note ``defect_count`` is the *gate* count (what triggered the run),
    # so it is not the denominator: a fit triggered by 1000 new defects may label
    # 5000. ``noise_count / labelled_count`` is the drift signal the monitor plots.
    labelled_count: int | None = None
    noise_count: int | None = None
    # HDBSCAN membership strength averaged over the run. Falling means points are
    # landing at the edges of clusters — drift, before it becomes outright noise.
    mean_probability: float | None = None
    error: str | None = None
    run_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: datetime = field(default_factory=_utcnow)
    finished_at: datetime = field(default_factory=_utcnow)

    @property
    def ok(self) -> bool:
        return self.status == "completed"


@dataclass(frozen=True)
class ClusterFit:
    """What one whole-set clustering produced: the labels *and* the fitted state.

    ``state`` is the serialised bundle that makes the clustering reusable — the
    normal prototype, the per-block scalers, PCA, UMAP, the HDBSCAN clusterer and
    the second-layer split decision. Persisting it is what lets later batches be
    *assigned* to these same clusters instead of re-fitting, so a ``cluster_id``
    keeps its meaning. ``None`` means the model cannot be frozen, and every run
    against it has to be a full re-fit.
    """

    assignments: list[ClusterAssignment]
    state: bytes | None = None

    @property
    def cluster_count(self) -> int:
        """Distinct clusters found, excluding ``-1`` noise."""
        return len({a.cluster_id for a in self.assignments if a.cluster_id != -1})
