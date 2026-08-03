"""Postgres adapters, one module per table.

:mod:`~adapters.outbound.db.repository` holds the infrastructure every table
adapter shares — the :class:`Database` that owns the process-wide connection
pool and the transaction/rollback contract. The other modules are named for the
table (and so the entity) they read or write:

* :mod:`~adapters.outbound.db.models`            — ``inference_model``
* :mod:`~adapters.outbound.db.inference_results` — ``inference_results``

The composition root builds one :class:`Database` and injects it into every
repository, so the service holds a single pool and closes it once.
"""
from adapters.outbound.db.inference_results import PostgresImageRepository
from adapters.outbound.db.models import PostgresModelRegistry
from adapters.outbound.db.repository import Database, PostgresRepository

__all__ = [
    "Database",
    "PostgresRepository",
    "PostgresModelRegistry",
    "PostgresImageRepository",
]
