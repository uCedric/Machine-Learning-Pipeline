"""DINOv2 builder: assemble a ready-to-run batch clusterer from a model version.

Registered with the service-wide :class:`~adapters.outbound.factory.ModelFactory`.
Like the ResNet50 clusterer, the DINOv2 backbone is a fixed pretrained network
(loaded from torch hub, no per-version assets to fetch), so the builder does not
read object storage; the registry row it is handed gates validity/version. The
clusterer carries ``version.model_id`` for result attribution.
"""
from __future__ import annotations

from adapters.outbound.dinov2.model import Dinov2ClusterModel
from adapters.outbound.dinov2.resources import load_resources
from application.ports.storage import ObjectStorage
from config.settings import ModelConfig
from domain.models import ModelVersion


def build_dinov2(
    version: ModelVersion, storage: ObjectStorage, config: ModelConfig
) -> Dinov2ClusterModel:
    resources = load_resources(config.image_size)
    return Dinov2ClusterModel(
        resources.model,
        resources.layers,
        resources.layer_names,
        resources.transform,
        resources.device,
        resources.input_size,
        model_id=version.model_id,
    )
