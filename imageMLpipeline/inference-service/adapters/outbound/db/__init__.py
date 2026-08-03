"""Postgres adapters, one module per table.

:mod:`~adapters.outbound.db.repository` holds the infrastructure every table
adapter shares — the :class:`Database` that owns the process-wide connection
pool and the transaction/rollback contract. The other modules are named for the
table (and so the entity) they persist:

* :mod:`~adapters.outbound.db.images`            — ``image``
* :mod:`~adapters.outbound.db.models`            — ``inference_model``
* :mod:`~adapters.outbound.db.inference_results` — ``inference_results``
* :mod:`~adapters.outbound.db.cluster_results`   — ``cluster_results``
* :mod:`~adapters.outbound.db.stage_two_runs`    — ``stage_two_run``

The composition root builds one :class:`Database` and injects it into every
repository, so the service holds a single pool and closes it once.
"""
from adapters.outbound.db.cluster_results import PostgresClusterResultRepository
from adapters.outbound.db.images import PostgresImageRepository
from adapters.outbound.db.inference_results import (
    PostgresClusteringImageSource,
    PostgresResultRepository,
)
from adapters.outbound.db.models import PostgresModelRegistry
from adapters.outbound.db.repository import Database, PostgresRepository
from adapters.outbound.db.stage_two_runs import PostgresClusteringRunLog

__all__ = [
    "Database",
    "PostgresRepository",
    "PostgresImageRepository",
    "PostgresModelRegistry",
    "PostgresResultRepository",
    "PostgresClusteringImageSource",
    "PostgresClusterResultRepository",
    "PostgresClusteringRunLog",
]
