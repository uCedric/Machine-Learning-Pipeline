"""Training use case.

Pure application logic. For now it just acknowledges receipt of a train event;
the actual training workflow will be added later.
"""
from __future__ import annotations

import logging

from domain.models import TrainEvent

logger = logging.getLogger(__name__)


class RunTraining:
    def execute(self, event: TrainEvent) -> None:
        logger.info("Received train event: %s", event.payload)
