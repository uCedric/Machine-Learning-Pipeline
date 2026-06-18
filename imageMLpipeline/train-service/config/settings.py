"""Typed configuration assembled from environment variables (12-factor).

All environment reads live here so the rest of the codebase stays free of
``os.getenv`` calls. Configuration is read once at the edge (the composition root
in :mod:`bootstrap`) and injected inward.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str
    topic: str
    group_id: str
    event_type: str


@dataclass(frozen=True)
class Settings:
    kafka: KafkaConfig

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            kafka=KafkaConfig(
                bootstrap_servers=_env("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"),
                topic=_env("KAFKA_TOPIC", "train"),
                group_id=_env("KAFKA_GROUP_ID", "train-service"),
                event_type=_env("KAFKA_EVENT_TYPE", "train"),
            ),
        )
