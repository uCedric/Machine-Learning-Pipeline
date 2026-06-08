"""Typed configuration assembled from environment variables (12-factor).

All environment reads live here so the rest of the codebase stays free of
``os.getenv`` calls and is easy to test. Configuration is read once at the edge
(the composition roots in :mod:`bootstrap`) and injected inward.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Default PatchCore asset locations inside the container image. The Docker build
# copies ``adapters/`` to ``/app/adapters``; ``minio-init`` also uploads the same
# assets to the ``models`` bucket for runtime distribution.
_DEFAULT_MEMORY_BANK_PATH = "/app/adapters/outbound/patchcore/asset/memory_bank.npy"
_DEFAULT_ONNX_PATH = "/app/adapters/outbound/patchcore/asset/resnet_backbone.onnx"
_DEFAULT_THRESHOLD = "16.315967559814453"


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _bool(key: str, default: str = "false") -> bool:
    return _env(key, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str
    topic: str
    group_id: str


@dataclass(frozen=True)
class MinioConfig:
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool
    images_bucket: str


@dataclass(frozen=True)
class IcebergConfig:
    catalog_name: str
    namespace: str
    table: str
    warehouse: str
    sql_uri: str
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    s3_region: str


@dataclass(frozen=True)
class EltConfig:
    input_file: str
    processed_dir: str
    poll_interval: float
    run_once: bool


@dataclass(frozen=True)
class ModelConfig:
    name: str
    memory_bank_path: str
    onnx_path: str
    image_size: int
    threshold: float


@dataclass(frozen=True)
class Settings:
    kafka: KafkaConfig
    minio: MinioConfig
    iceberg: IcebergConfig
    elt: EltConfig
    model: ModelConfig

    @staticmethod
    def from_env() -> "Settings":
        endpoint = _env("MINIO_ENDPOINT", "minio:9000")
        access_key = _env("MINIO_ACCESS_KEY", _env("MINIO_ROOT_USER", "minioadmin"))
        secret_key = _env("MINIO_SECRET_KEY", _env("MINIO_ROOT_PASSWORD", "minioadmin"))
        secure = _bool("MINIO_SECURE", "false")
        warehouse_bucket = _env("WAREHOUSE_BUCKET", "warehouse")
        scheme = "https" if secure else "http"

        return Settings(
            kafka=KafkaConfig(
                bootstrap_servers=_env("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"),
                topic=_env("KAFKA_TOPIC", "inference-events"),
                group_id=_env("KAFKA_GROUP_ID", "inference-service"),
            ),
            minio=MinioConfig(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
                images_bucket=_env("IMAGES_BUCKET", "images"),
            ),
            iceberg=IcebergConfig(
                catalog_name=_env("ICEBERG_CATALOG_NAME", "demo"),
                namespace=_env("ICEBERG_NAMESPACE", "inference"),
                table=_env("ICEBERG_TABLE", "results"),
                warehouse=_env("ICEBERG_WAREHOUSE", f"s3://{warehouse_bucket}/"),
                sql_uri=_env("ICEBERG_SQL_URI", _postgres_uri()),
                s3_endpoint=_env("ICEBERG_S3_ENDPOINT", f"{scheme}://{endpoint}"),
                s3_access_key=access_key,
                s3_secret_key=secret_key,
                s3_region=_env("AWS_REGION", "us-east-1"),
            ),
            elt=EltConfig(
                input_file=_env("ELT_INPUT_FILE", "/data/007.png"),
                processed_dir=_env("ELT_PROCESSED_DIR", "/data/processed"),
                poll_interval=float(_env("ELT_POLL_INTERVAL", "5")),
                run_once=_bool("ELT_RUN_ONCE", "true"),
            ),
            model=ModelConfig(
                name=_env("MODEL_NAME", "mock-model-v1"),
                memory_bank_path=_env("MODEL_MEMORY_BANK_PATH", _DEFAULT_MEMORY_BANK_PATH),
                onnx_path=_env("MODEL_ONNX_PATH", _DEFAULT_ONNX_PATH),
                image_size=int(_env("MODEL_IMAGE_SIZE", "224")),
                threshold=float(_env("MODEL_THRESHOLD", _DEFAULT_THRESHOLD)),
            ),
        )


def _postgres_uri() -> str:
    user = _env("POSTGRES_USER", "postgres")
    password = _env("POSTGRES_PASSWORD", "postgres")
    host = _env("POSTGRES_HOST", "postgres")
    port = _env("POSTGRES_PORT", "5432")
    database = _env("POSTGRES_DB", "postgres")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
