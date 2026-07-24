"""MinIO-backed memory-bank repository for PatchCore.

Stores the memory bank ``.npy`` and copies the unchanged backbone / buffer-zone
assets between version directories (``{model_type}/{version}/``) via the object
storage port.
"""
from __future__ import annotations

import logging

import numpy as np

from adapters.outbound.patchcore.resources import (
    ASSET_BUFFER_ZONE,
    ASSET_MEMORY_BANK,
    UNCHANGED_ASSETS,
    load_memory_bank,
    prefix_for,
    serialize_buffer_zone,
    serialize_memory_bank,
)
from application.ports.memory_bank import MemoryBankRepository
from application.ports.storage import ObjectStorage
from domain.models import ModelVersion

logger = logging.getLogger(__name__)


def _content_type(name: str) -> str:
    return "text/plain" if name.endswith(".txt") else "application/octet-stream"


class PatchCoreMemoryBankStore(MemoryBankRepository):
    def __init__(self, storage: ObjectStorage) -> None:
        self._storage = storage

    def load(self, version: ModelVersion) -> np.ndarray:
        return load_memory_bank(
            self._storage, version.bucket, version.model_type, version.version
        )

    def save(self, version: ModelVersion, bank: np.ndarray) -> None:
        prefix = prefix_for(version.model_type, version.version)
        key = f"{prefix}/{ASSET_MEMORY_BANK}"
        self._storage.put_object(
            version.bucket, key, serialize_memory_bank(bank), "application/octet-stream"
        )
        logger.info("Saved memory bank %s to %s/%s", bank.shape, version.bucket, key)

    def save_buffer_zone(
        self, version: ModelVersion, lower: float, upper: float
    ) -> None:
        prefix = prefix_for(version.model_type, version.version)
        key = f"{prefix}/{ASSET_BUFFER_ZONE}"
        self._storage.put_object(
            version.bucket, key, serialize_buffer_zone(lower, upper), "text/plain"
        )
        logger.info(
            "Saved buffer zone lower=%s upper=%s to %s/%s",
            lower,
            upper,
            version.bucket,
            key,
        )

    def copy_unchanged_assets(self, src: ModelVersion, dst: ModelVersion) -> None:
        for name in UNCHANGED_ASSETS:
            self._copy_asset(src, dst, name)

    def copy_buffer_zone(self, src: ModelVersion, dst: ModelVersion) -> None:
        self._copy_asset(src, dst, ASSET_BUFFER_ZONE)

    def _copy_asset(self, src: ModelVersion, dst: ModelVersion, name: str) -> None:
        src_prefix = prefix_for(src.model_type, src.version)
        dst_prefix = prefix_for(dst.model_type, dst.version)
        data = self._storage.get_object(src.bucket, f"{src_prefix}/{name}")
        self._storage.put_object(
            dst.bucket, f"{dst_prefix}/{name}", data, _content_type(name)
        )
        logger.info(
            "Copied asset %s: %s/%s -> %s/%s",
            name,
            src.bucket,
            src_prefix,
            dst.bucket,
            dst_prefix,
        )
