"""Composite ResultRepository: fan a single result out to several repositories.

The inference pipeline persists every result to both Iceberg (analytics
warehouse) and Postgres (directly SQL-queryable). Modelling that as one
:class:`~application.ports.repository.ResultRepository` keeps the use case
unaware that there is more than one sink: it calls ``save`` once.
"""
from __future__ import annotations

import logging

from application.ports.repository import ResultRepository
from domain.models import InferenceResult

logger = logging.getLogger(__name__)


class CompositeResultRepository(ResultRepository):
    def __init__(self, *repositories: ResultRepository) -> None:
        self._repositories = repositories

    def save(self, result: InferenceResult) -> None:
        for repository in self._repositories:
            repository.save(result)

    def close(self) -> None:
        for repository in self._repositories:
            close = getattr(repository, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.exception("Failed to close %s", type(repository).__name__)
