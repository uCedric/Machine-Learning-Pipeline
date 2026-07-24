"""DINOv2 builder: assemble a ready-to-run clusterer from a model version.

Registered with the service-wide :class:`~adapters.outbound.factory.ModelFactory`.
Holds the DINOv2-specific knowledge — which assets the clusterer needs and how
to load them — so the factory can stay model-agnostic.
"""
from __future__ import annotations

from adapters.outbound.dinov2.model import Dinov2ClusterModel
from adapters.outbound.dinov2.resources import fetch_assets, load_resources
from application.ports.storage import ObjectStorage
from config.settings import ModelConfig
from domain.models import ModelVersion


def build_dinov2(
    version: ModelVersion, storage: ObjectStorage, config: ModelConfig
) -> Dinov2ClusterModel:
    """Fetch the version's assets from object storage and build the clusterer.

    Assets live in ``version.bucket`` under ``{model_type}/{version}/`` and are
    cached on disk before loading, so the model is distributed at runtime rather
    than baked into the image. The clusterer carries ``version.model_id`` for
    result attribution.
    """
    assets = fetch_assets(
        storage, version.bucket, version.model_type, version.version, config.cache_dir
    )
    resources = load_resources(assets, config.image_size)
    return Dinov2ClusterModel(
        resources.session,
        resources.pca,
        resources.umap,
        resources.clusterer,
        resources.transform,
        model_id=version.model_id,
    )
