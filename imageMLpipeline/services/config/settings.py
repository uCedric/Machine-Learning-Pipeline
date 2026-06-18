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
_DEFAULT_BUFFER_ZONE_PATH = "/app/adapters/outbound/patchcore/asset/buffer_zone.txt"
# Default model registry id. Must match the seeded row in migrations/V1__init.sql.
_DEFAULT_MODEL_ID = "2526d2bd-0725-43a2-a29b-4905fb0e5335"


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _required(*keys: str) -> str:
    """Return the first non-empty value among ``keys``.

    Credentials have no safe default and must be supplied via the environment
    (loaded from ``.env``); raise if none of ``keys`` is set.
    """
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    raise RuntimeError(f"Missing required credential; set one of: {', '.join(keys)}")


def _bool(key: str, default: str = "false") -> bool:
    return _env(key, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str
    topic: str
    group_id: str
    event_type: str


@dataclass(frozen=True)
class MinioConfig:
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool
    images_bucket: str


@dataclass(frozen=True)
class PostgresConfig:
    sql_uri: str


@dataclass(frozen=True)
class EltConfig:
    input_folder: str
    processed_dir: str
    poll_interval: float
    run_once: bool


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    memory_bank_path: str
    onnx_path: str
    image_size: int
    buffer_zone_path: str


@dataclass(frozen=True)
class Settings:
    kafka: KafkaConfig
    minio: MinioConfig
    postgres: PostgresConfig
    elt: EltConfig
    model: ModelConfig

    @staticmethod
    def from_env() -> "Settings":
        endpoint = _env("MINIO_ENDPOINT", "minio:9000")
        access_key = _required("MINIO_ACCESS_KEY", "MINIO_ROOT_USER")
        secret_key = _required("MINIO_SECRET_KEY", "MINIO_ROOT_PASSWORD")
        secure = _bool("MINIO_SECURE", "false")

        return Settings(
            kafka=KafkaConfig(
                bootstrap_servers=_env("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"),
                topic=_env("KAFKA_TOPIC", "inference"),
                group_id=_env("KAFKA_GROUP_ID", "inference-service"),
                event_type=_env("KAFKA_EVENT_TYPE", "inference"),
            ),
            minio=MinioConfig(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
                images_bucket=_env("IMAGES_BUCKET", "images"),
            ),
            postgres=PostgresConfig(
                sql_uri=_env("POSTGRES_SQL_URI", _postgres_uri()),
            ),
            elt=EltConfig(
                input_folder=_env("ELT_INPUT_FOLDER", "/data"),
                processed_dir=_env("ELT_PROCESSED_DIR", "/data/processed"),
                poll_interval=float(_env("ELT_POLL_INTERVAL", "5")),
                run_once=_bool("ELT_RUN_ONCE", "true"),
            ),
            model=ModelConfig(
                name=_env("MODEL_NAME", "patchcore"),
                model_id=_env("MODEL_ID", _DEFAULT_MODEL_ID),
                memory_bank_path=_env("MODEL_MEMORY_BANK_PATH", _DEFAULT_MEMORY_BANK_PATH),
                onnx_path=_env("MODEL_ONNX_PATH", _DEFAULT_ONNX_PATH),
                image_size=int(_env("MODEL_IMAGE_SIZE", "224")),
                buffer_zone_path=_env("MODEL_BUFFER_ZONE_PATH", _DEFAULT_BUFFER_ZONE_PATH),
            ),
        )


def _postgres_uri() -> str:
    user = _required("POSTGRES_USER")
    password = _required("POSTGRES_PASSWORD")
    host = _env("POSTGRES_HOST", "postgres")
    port = _env("POSTGRES_PORT", "5432")
    database = _env("POSTGRES_DB", "postgres")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
