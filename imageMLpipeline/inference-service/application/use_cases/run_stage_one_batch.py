"""Stage-one over a batch of events: score them, then hand defects to stage-two.

Pure application logic wired against :mod:`application.ports`: the batch scorer
and the event publisher. It holds no reference to Spark, MinIO, ONNX or
Postgres — the composition root injects a concrete
:class:`~application.ports.batch_scorer.BatchAnomalyScorer`, which today is the
Spark adapter but could equally be an in-process loop.

The scorer persists each result where the image was scored (see that port's
docstring for why). What is left for this use case is the part that must happen
*once per batch, in one place*: re-publishing the defect and uncertain images as
``stage-two`` triggers.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Collection, Sequence

from application.ports.batch_scorer import BatchAnomalyScorer, ScoredEvent
from application.ports.events import EventPublisher
from domain.models import InferenceEvent, StageType

logger = logging.getLogger(__name__)


class RunStageOneBatch:
    def __init__(
        self,
        scorer: BatchAnomalyScorer,
        *,
        publisher: EventPublisher | None = None,
        topic: str = "inference",
        trigger_statuses: Collection[str] = ("anomaly",),
    ) -> None:
        self._scorer = scorer
        self._publisher = publisher
        # Stage-two re-uses the same 'inference' topic; only the event's 'type'
        # changes (to stage-two).
        self._topic = topic
        # Which verdicts hand an image on to stage-two clustering: confirmed
        # defects only. The uncertain 'pending' band goes to human review in the
        # monitor dashboard instead, so triggering on it would produce events the
        # stage-two gate cannot count. A frozenset for O(1) membership tests.
        self._trigger_statuses = frozenset(trigger_statuses)

    def execute(self, events: Sequence[InferenceEvent]) -> list[ScoredEvent]:
        if not events:
            return []

        scored = self._scorer.score(events)
        for result in scored:
            if result.ok and result.status in self._trigger_statuses:
                self._emit_stage_two(result.event)
        return scored

    def _emit_stage_two(self, event: InferenceEvent) -> None:
        """Re-publish one image as a stage-two trigger.

        A publish failure must never fail the already-saved anomaly result, so
        it is logged and swallowed — exactly as the per-event path does.
        """
        if self._publisher is None:
            return
        stage_two_event = replace(event, type=StageType.STAGE_TWO)
        try:
            self._publisher.publish(self._topic, stage_two_event)
            logger.info(
                "Published stage-two trigger for %s/%s",
                stage_two_event.bucket,
                stage_two_event.object_key,
            )
        except Exception:
            logger.exception(
                "Failed to publish stage-two trigger for %s/%s; inference result already saved",
                event.bucket,
                event.object_key,
            )
