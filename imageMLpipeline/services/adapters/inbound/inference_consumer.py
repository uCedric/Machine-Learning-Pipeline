"""Inference driving adapter: consume events and run the inference use case.

Offsets are committed only after a result is saved (at-least-once delivery).
"""
from __future__ import annotations

import logging

from application.ports.events import EventConsumer
from application.use_cases.run_inference import RunInferenceUseCase

logger = logging.getLogger("inference-service")


class InferenceConsumer:
    def __init__(self, consumer: EventConsumer, use_case: RunInferenceUseCase) -> None:
        self._consumer = consumer
        self._use_case = use_case

    def run(self) -> None:
        for event in self._consumer.events():
            try:
                self._use_case.execute(event)
                self._consumer.commit()
            except Exception:
                logger.exception(
                    "Failed to process event %s for %s/%s",
                    event.event_id,
                    event.bucket,
                    event.object_key,
                )
