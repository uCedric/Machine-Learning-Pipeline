"""PatchCore builder: assemble a ready-to-run detector from a model version.

Registered with the service-wide :class:`~adapters.outbound.factory.ModelFactory`.
Holds the PatchCore-specific knowledge — which assets the model needs and how to
load them — so the factory can stay model-agnostic.
"""
from __future__ import annotations

from adapters.outbound.patchcore.model import PatchCore
from adapters.outbound.patchcore.resources import fetch_assets, load_resources
from application.ports.storage import ObjectStorage
from config.settings import ModelConfig
from domain.models import ModelVersion


def build_patchcore(
    version: ModelVersion, storage: ObjectStorage, config: ModelConfig
) -> PatchCore:
    """Fetch the version's assets from object storage and build the detector.

    Assets live in ``version.bucket`` under ``{model_type}/{version}/`` and are
    cached on disk before loading, so the model is distributed at runtime rather
    than baked into the image. The detector carries ``version.model_id`` for
    result attribution.
    """
    assets = fetch_assets(
        storage, version.bucket, version.model_type, version.version, config.cache_dir
    )
    resources = load_resources(
        assets.memory_bank, assets.onnx, assets.buffer_zone, config.image_size
    )
    return PatchCore(
        resources.session,
        resources.index,
        resources.transform,
        resources.buffer_zone,
        model_id=version.model_id,
    )
