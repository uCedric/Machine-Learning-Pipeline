"""Model factory (Factory design pattern).

Service-wide entry point for initialising ML models. A model type is requested
by key; the factory resolves its newest registered version from the model
registry and delegates to the builder registered for that type. The factory
itself is model-agnostic — concrete builders live with their model package
(e.g. :func:`adapters.outbound.patchcore.builder.build_patchcore`) and are
registered via :meth:`ModelFactory.register`. Adding a new model for the
business is therefore: write its builder, register it, done.
"""
from __future__ import annotations

import logging
from typing import Callable

from adapters.outbound.dinov2.builder import build_dinov2
from adapters.outbound.patchcore.builder import build_patchcore
from adapters.outbound.resnet50.builder import build_resnet50
from application.ports.model import AnomalyModel, ClusterModel
from application.ports.model_registry import ModelRegistry
from application.ports.storage import ObjectStorage
from config.settings import ModelConfig
from domain.models import ModelVersion

logger = logging.getLogger(__name__)

# A model port implementation — anomaly detection or clustering.
Model = AnomalyModel | ClusterModel

# Assembles a resolved model version into a ready-to-run model. One builder per
# model type, defined alongside its model package so the factory stays generic.
ModelBuilder = Callable[[ModelVersion, ObjectStorage, ModelConfig], Model]


class ModelFactory:
    """Builds ready-to-run models by type (Factory pattern).

    Model-agnostic: it resolves the newest valid version of a requested type
    from the registry and dispatches to the builder registered for that type.
    The registry and object storage are shared dependencies injected once;
    register a new model type with :meth:`register`.
    """

    def __init__(self, registry: ModelRegistry, storage: ObjectStorage) -> None:
        self._registry = registry
        self._storage = storage
        self._builders: dict[str, ModelBuilder] = {}
        # Register the built-in model builders. Adding a new model type means
        # writing its builder and registering it here (or via :meth:`register`).
        # Stage-one anomaly: patchcore. Stage-two clustering: resnet50 (deployed)
        # and dinov2 (a candidate, registered is_valid=false until promoted).
        self.register("patchcore", build_patchcore)
        self.register("resnet50", build_resnet50)
        self.register("dinov2", build_dinov2)

    def register(self, model_type: str, builder: ModelBuilder) -> None:
        """Register the builder that assembles ``model_type``."""
        self._builders[model_type.lower()] = builder

    def build(self, config: ModelConfig) -> Model:
        """Build the newest registered version of ``config.model_key``.

        Resolves the latest valid version from the registry, then delegates to
        the builder registered for that model type.

        Raises:
            ValueError: if no builder is registered for the model type.
            LookupError: if the registry holds no valid version (propagated).
        """
        model_type = config.model_key
        builder = self._builders.get(model_type.lower())
        if builder is None:
            raise ValueError(
                f"No builder registered for model '{model_type}'. "
                f"Registered: {sorted(self._builders)}"
            )
        version = self._registry.latest(model_type)
        logger.info(
            "Building model '%s' v%s (model_id=%s)",
            version.model_type,
            version.version,
            version.model_id,
        )
        return builder(version, self._storage, config)
