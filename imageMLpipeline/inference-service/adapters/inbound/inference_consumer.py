"""Inference driving adapter: consume 'inference' events and dispatch by stage.

One service, one topic. Each ``inference`` event carries a ``type``
(:class:`~domain.models.StageType`); the consumer routes it:

* ``stage-one`` → :class:`RunStageOneBatch` (PatchCore anomaly detection over a
  micro-batch, scored on the Spark cluster); on an ANOMALY verdict it
  re-publishes the event tagged ``stage-two``.
* ``stage-two`` → :class:`RunStageTwo` (defect clustering: assign new defects to
  the frozen fit, or re-fit the whole set once enough have accumulated).
* ``refresh``   → :class:`RefreshUseCase` (re-read model status).

Stage-one is **batch-only**: events are accumulated into a micro-batch and
scored on the cluster, and there is no per-event in-process path. That is why no
anomaly model is built here — it lives in the Spark Python workers, one per
process, and this container never loads it. The clustering backbone, by
contrast, is built here, because stage-two runs in this process.

The use cases and their port implementations — object storage, the clustering
backbone, the repositories, the clustering image source, and the Kafka
consumer/publisher — are wired from :class:`Settings` via
:meth:`InferenceConsumer.initialize`, which delegates each stage to its own
``*_producer`` classmethod. Every Postgres adapter shares the one
:class:`Database` built here, so the service holds a single connection pool and
closes it once.

The clustering backbone is only built when a valid ``inference_model`` row exists
for ``CLUSTER_MODEL_KEY`` (default ``resnet50``); otherwise stage-two triggers are
logged and skipped. Offsets are committed only after an event is handled
(at-least-once delivery).
"""
from __future__ import annotations

import logging
from functools import partial
from typing import Protocol

from adapters.outbound.db import (
    Database,
    PostgresClusteringImageSource,
    PostgresClusteringRunLog,
    PostgresClusterResultRepository,
    PostgresModelRegistry,
)
from adapters.outbound.factory import ModelFactory
from adapters.outbound.kafka_events import KafkaEventConsumer, KafkaEventPublisher
from adapters.outbound.minio_storage import MinioObjectStorage
from application.ports.events import EventConsumer
from application.ports.model import ClusterModel
from application.use_cases.refresh import RefreshUseCase
from application.use_cases.run_stage_one_batch import RunStageOneBatch
from application.use_cases.run_stage_two import RunStageTwo
from config.settings import Settings
from domain.models import InferenceEvent, StageType

logger = logging.getLogger("inference-service")


class _Closeable(Protocol):
    def close(self) -> None: ...


class InferenceConsumer:
    def __init__(
        self,
        consumer: EventConsumer,
        stage_one: RunStageOneBatch,
        stage_two: RunStageTwo | None,
        refresh: RefreshUseCase,
        *,
        micro_batch_size: int = 32,
        poll_timeout_ms: int = 2000,
        closables: tuple[_Closeable, ...] = (),
    ) -> None:
        self._consumer = consumer
        # Stage-one, scored on the cluster. There is no in-process alternative:
        # one code path, one place where a verdict can come from.
        self._stage_one = stage_one
        self._stage_two = stage_two
        self._refresh = refresh
        self._micro_batch_size = micro_batch_size
        self._poll_timeout_ms = poll_timeout_ms
        self._closables = closables

    @classmethod
    def cluster_model_producer(
        cls,
        settings: Settings,
        storage: MinioObjectStorage,
        database: Database,
    ) -> ClusterModel | None:
        """Resolve and build the stage-two backbone from the model registry.

        Optional: a ``LookupError`` means no valid version is registered, so
        clustering stays disabled and the service runs stage-one only.

        The stage-one anomaly model is deliberately absent — it is built inside
        each Spark Python worker, so building one here would load ~45 MB of
        weights this process never uses.
        """
        registry = PostgresModelRegistry(database)
        factory = ModelFactory(registry, storage)
        try:
            return factory.build(settings.cluster_model)
        except LookupError:
            logger.warning(
                "No valid '%s' clustering backbone registered; stage-two "
                "triggers will be skipped",
                settings.cluster_model.model_key,
            )
            return None

    @classmethod
    def stage_two_producer(
        cls,
        settings: Settings,
        cluster_model: ClusterModel,
        database: Database,
        storage: MinioObjectStorage,
    ) -> RunStageTwo:
        from adapters.outbound.resnet50.resources import save_state

        return RunStageTwo(
            cluster_model,
            PostgresClusteringImageSource(database),
            PostgresClusterResultRepository(database),
            PostgresClusteringRunLog(database),
            PostgresModelRegistry(database),
            storage_get=storage.get_object,
            # A fit freezes itself as a new version's artifact; the key layout
            # belongs to the model package, so the root only passes the callable.
            state_put=partial(save_state, storage),
            model_id=cluster_model.model_id,
            model_key=settings.cluster_model.model_key,
            # Known-good images come from the stage-one anomaly model's verdicts.
            normal_model_type=settings.model.model_key,
            min_batch=settings.stage_two.min_batch,
            refit_count=settings.stage_two.refit_count,
            assign_count=settings.stage_two.assign_count,
            normal_set_size=settings.stage_two.normal_set_size,
            defect_set_size=settings.stage_two.defect_set_size,
        )

    @classmethod
    def stage_one_producer(
        cls,
        settings: Settings,
        publisher: KafkaEventPublisher,
    ) -> "tuple[RunStageOneBatch, _Closeable]":
        """Build the stage-one path: a batch use case over the Spark scorer.

        Returns the use case together with the Spark session to close, so the
        composition root owns that lifecycle rather than reaching into the use
        case for it.

        Reaching the cluster is a hard requirement — there is no in-process
        fallback to degrade to, so a failure here stops the service at start-up
        rather than letting it run and silently score nothing.
        """
        from adapters.outbound.spark import SparkAnomalyScorer

        scorer = SparkAnomalyScorer.initialize(settings)
        use_case = RunStageOneBatch(
            scorer,
            publisher=publisher,
            topic=settings.kafka.topic,
            trigger_statuses=settings.stage_two.trigger_statuses,
        )
        return use_case, scorer

    @classmethod
    def refresh_producer(cls, stage_two: RunStageTwo | None) -> RefreshUseCase:
        return RefreshUseCase(stage_two)

    @classmethod
    def initialize(cls, settings: Settings) -> "InferenceConsumer":
        """Build every stage's port implementations from configuration and wire them.

        Stage-one needs no model here: it is scored on the cluster, so only the
        Spark session is built. The clustering backbone (stage-two) is optional —
        no valid registered version means clustering stays off and stage-two
        triggers are logged and skipped until an operator promotes one.
        """
        storage = MinioObjectStorage(settings.minio)
        database = Database(settings.postgres.sql_uri)
        publisher = KafkaEventPublisher(settings.kafka)

        # ── Stage-one: anomaly detection on the cluster (+ stage-two triggers) ──
        stage_one, spark_session = cls.stage_one_producer(settings, publisher)

        # ── Stage-two: batch defect clustering (only when a backbone is valid) ──
        cluster_model = cls.cluster_model_producer(settings, storage, database)
        stage_two = (
            cls.stage_two_producer(settings, cluster_model, database, storage)
            if cluster_model is not None
            else None
        )

        # ── Refresh ML model status ──
        refresh = cls.refresh_producer(stage_two)

        consumer = KafkaEventConsumer(settings.kafka)
        # The repositories hold no resource of their own — the pool belongs to
        # the database — so closing it once releases all of them.
        closables: tuple[_Closeable, ...] = (
            consumer,
            publisher,
            database,
            spark_session,
        )

        return cls(
            consumer,
            stage_one,
            stage_two,
            refresh,
            micro_batch_size=settings.spark.micro_batch_size,
            poll_timeout_ms=settings.spark.poll_timeout_ms,
            closables=closables,
        )

    def run(self) -> None:
        """Poll a micro-batch, dispatch stage-one to Spark, handle the rest inline.

        Stage-one events are accumulated rather than handled one at a time:
        scoring runs on the Spark cluster, and a round-trip per image would cost
        far more in scheduling than the ~1s of work it distributes. Stage-two and
        refresh stay per-event — stage-two is already a whole-set batch of its own.
        """
        while True:
            events = self._consumer.poll(self._micro_batch_size, self._poll_timeout_ms)
            if not events:
                continue

            stage_one_events = [e for e in events if e.type == StageType.STAGE_ONE]
            others = [e for e in events if e.type != StageType.STAGE_ONE]

            try:
                self._handle_stage_one(stage_one_events)
                for event in others:
                    self._handle_other(event)
                # One commit per micro-batch: at-least-once still holds, and a
                # crash mid-batch replays the whole batch rather than losing it.
                self._consumer.commit()
            except Exception:
                logger.exception("Failed to process a batch of %d event(s)", len(events))

    def _handle_stage_one(self, events: list[InferenceEvent]) -> None:
        if not events:
            return
        self._stage_one.execute(events)

    def _handle_other(self, event: InferenceEvent) -> None:
        if event.type == StageType.STAGE_TWO:
            if self._stage_two is not None:
                self._stage_two.execute(event)
            else:
                logger.info(
                    "stage-two clustering disabled; skipping trigger for %s/%s",
                    event.bucket,
                    event.object_key,
                )
        elif event.type == StageType.REFRESH:
            self._refresh.execute(event)

    def close(self) -> None:
        """Release the owned port implementations."""
        for closable in self._closables:
            closable.close()
