"""Postgres access for the dashboard, one module per table.

:mod:`db.repository` holds the :class:`Database` that owns the connection pool
and the transaction/rollback contract; the other modules hold the named queries
the HTTP endpoints call:

* :mod:`db.inference_results` — stage-one verdicts.
* :mod:`db.stage_two_runs`    — stage-two clustering runs and their drift signal.
"""
from db.inference_results import InferenceResultsQueries
from db.repository import Database, PostgresRepository
from db.stage_two_runs import StageTwoRunQueries

__all__ = [
    "Database",
    "PostgresRepository",
    "InferenceResultsQueries",
    "StageTwoRunQueries",
]
