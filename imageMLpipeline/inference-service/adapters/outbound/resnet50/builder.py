"""ResNet50 builder: assemble a ready-to-run clusterer from a model version.

Registered with the service-wide :class:`~adapters.outbound.factory.ModelFactory`.
The *backbone* is a fixed pretrained network with no per-version weights, but the
*fit* is a per-version artifact: if this version has a frozen
``cluster_state.joblib`` in object storage, it is loaded so the clusterer can
**assign** new images to that version's clusters. Without one the clusterer can
only **fit**, which is exactly the state of a freshly seeded version.

The clusterer carries ``version.model_id`` for result attribution.
"""
from __future__ import annotations

import logging

from adapters.outbound.resnet50.model import ResNet50ClusterModel
from adapters.outbound.resnet50.resources import load_resources, load_state, state_key
from application.ports.storage import ObjectStorage
from config.settings import ModelConfig
from domain.models import ModelVersion

logger = logging.getLogger(__name__)


def build_resnet50(
    version: ModelVersion, storage: ObjectStorage, config: ModelConfig
) -> ResNet50ClusterModel:
    resources = load_resources(config.image_size)
    return ResNet50ClusterModel(
        resources.feature_extractor,
        resources.transform,
        resources.device,
        image_size=config.image_size,
        model_id=version.model_id,
        state=_load_frozen_fit(version, storage, resources.device),
    )


def _load_frozen_fit(version: ModelVersion, storage: ObjectStorage, device: str):
    """Fetch this version's frozen fit, or ``None`` if it has never been fitted.

    A missing artifact is the normal bootstrap case — the version seeded by the
    migrations has no fit yet — so it is reported and swallowed rather than raised.
    The :class:`ObjectStorage` port has no ``exists``, and adding one just for this
    would push an SDK concern inward, so absence is detected by the read failing.
    The exception type is logged so a genuine storage outage stays distinguishable
    from "not fitted yet".
    """
    key = state_key(version.model_type, version.version)
    try:
        data = storage.get_object(version.bucket, key)
    except Exception as exc:  # noqa: BLE001 - absence and outage look alike here
        logger.info(
            "No frozen fit at %s/%s (%s); this version can fit but not assign",
            version.bucket,
            key,
            type(exc).__name__,
        )
        return None
    state = load_state(data, device=device)
    logger.info(
        "Loaded frozen fit %s/%s (%d bytes); assign is available",
        version.bucket,
        key,
        len(data),
    )
    return state
