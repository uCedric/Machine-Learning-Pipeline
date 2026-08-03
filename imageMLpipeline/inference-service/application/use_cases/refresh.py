"""Refresh: re-read model status after an operator promotes a version.

Only stage-two is reachable from here. Stage-one's model no longer lives in this
process — it is built inside each Spark Python worker and cached there for the
life of that process — so a ``refresh`` event cannot reach it. Picking up a newly
promoted stage-one version currently means restarting the Spark workers; see
:mod:`adapters.outbound.spark.scorer`.
"""
from __future__ import annotations

import logging

from application.use_cases.run_stage_two import RunStageTwo
from domain.models import InferenceEvent

logger = logging.getLogger(__name__)


class RefreshUseCase:
    def __init__(self, stage_two: RunStageTwo | None) -> None:
        self._stage_two = stage_two

    def execute(self, event: InferenceEvent) -> None:
        logger.info(
            "Refresh requested (%s/%s); stage-one models live in the Spark "
            "workers and are not refreshed from here",
            event.bucket,
            event.object_key,
        )
