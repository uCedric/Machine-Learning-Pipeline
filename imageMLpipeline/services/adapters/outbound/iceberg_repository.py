"""Apache Iceberg adapter implementing the ResultRepository port.

Catalog : PostgreSQL (PyIceberg SqlCatalog).
Storage : MinIO (S3-compatible warehouse).

The namespace and table are created on first use, so the service is
self-bootstrapping against an empty catalog.
"""
from __future__ import annotations

import logging

import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError
from pyiceberg.schema import Schema
from pyiceberg.types import BooleanType, DoubleType, NestedField, StringType, TimestamptzType

from application.ports.repository import ResultRepository
from config.settings import IcebergConfig
from domain.models import InferenceResult

logger = logging.getLogger(__name__)

_SCHEMA = Schema(
    NestedField(1, "bucket", StringType(), required=True),
    NestedField(2, "object_key", StringType(), required=True),
    NestedField(3, "anomaly_score", DoubleType(), required=True),
    NestedField(4, "is_anomaly", BooleanType(), required=True),
    NestedField(5, "model_name", StringType(), required=True),
    NestedField(6, "heatmap_key", StringType(), required=True),
    NestedField(7, "inferred_at", TimestamptzType(), required=True),
)


class IcebergResultRepository(ResultRepository):
    def __init__(self, config: IcebergConfig) -> None:
        self._catalog = SqlCatalog(
            config.catalog_name,
            **{
                "uri": config.sql_uri,
                "warehouse": config.warehouse,
                "s3.endpoint": config.s3_endpoint,
                "s3.access-key-id": config.s3_access_key,
                "s3.secret-access-key": config.s3_secret_key,
                "s3.path-style-access": "true",
                "s3.region": config.s3_region,
            },
        )
        # Create PyIceberg's own catalog tables in Postgres if they don't exist.
        if hasattr(self._catalog, "create_tables"):
            self._catalog.create_tables()

        self._identifier = (config.namespace, config.table)
        self._table = self._ensure_table(config.namespace)

    def _ensure_table(self, namespace: str):
        try:
            self._catalog.create_namespace(namespace)
            logger.info("Created namespace '%s'", namespace)
        except NamespaceAlreadyExistsError:
            pass

        try:
            return self._catalog.load_table(self._identifier)
        except NoSuchTableError:
            logger.info("Creating Iceberg table %s", ".".join(self._identifier))
            return self._catalog.create_table(self._identifier, schema=_SCHEMA)

    def save(self, result: InferenceResult) -> None:
        record = pa.Table.from_pylist(
            [
                {
                    "bucket": result.bucket,
                    "object_key": result.object_key,
                    "anomaly_score": result.anomaly_score,
                    "is_anomaly": result.is_anomaly,
                    "model_name": result.model_name,
                    "heatmap_key": result.heatmap_key,
                    "inferred_at": result.inferred_at,
                }
            ],
            schema=self._table.schema().as_arrow(),
        )
        self._table.append(record)
