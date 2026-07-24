"""Typed configuration assembled from environment variables (12-factor).

All environment reads live here so the rest of the codebase stays free of
``os.getenv`` calls. Configuration is read once at the edge (the composition root
in :mod:`bootstrap`) and injected inward.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Local directory the model assets are downloaded to before loading. The bucket,
# model_type and version are resolved at runtime from the ``inference_model``
# table (the model registry), and the assets fetched from object storage rather
# than baked into the image.
_DEFAULT_MODEL_CACHE_DIR = "/tmp/model_assets"


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
class ModelConfig:
    name: str
    model_key: str
    image_size: int
    cache_dir: str
    # How many recent known-good images to score when recalibrating the buffer
    # zone after a retrain (the rolling validation set).
    validation_set_size: int


@dataclass(frozen=True)
class Settings:
    kafka: KafkaConfig
    minio: MinioConfig
    postgres: PostgresConfig
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
                topic=_env("KAFKA_TOPIC", "train"),
                group_id=_env("KAFKA_GROUP_ID", "train-service"),
                event_type=_env("KAFKA_EVENT_TYPE", "train"),
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
            model=ModelConfig(
                name=_env("MODEL_NAME", "patchcore"),
                model_key=_env("MODEL_KEY", "patchcore"),
                image_size=int(_env("MODEL_IMAGE_SIZE", "224")),
                cache_dir=_env("MODEL_CACHE_DIR", _DEFAULT_MODEL_CACHE_DIR),
                validation_set_size=int(_env("VALIDATION_SET_SIZE", "200")),
            ),
        )


def _postgres_uri() -> str:
    user = _required("POSTGRES_USER")
    password = _required("POSTGRES_PASSWORD")
    host = _env("POSTGRES_HOST", "postgres")
    port = _env("POSTGRES_PORT", "5432")
    database = _env("POSTGRES_DB", "postgres")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
