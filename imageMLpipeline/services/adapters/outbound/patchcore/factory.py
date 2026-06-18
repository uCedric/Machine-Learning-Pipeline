"""Model factory (Factory design pattern).

:class:`ModelFactory` creates :class:`~application.ports.model.AnomalyModel`
instances by name. PatchCore is the worked example — building it loads the FAISS
memory bank and the ONNX backbone. Register another model with
:meth:`ModelFactory.register`.
"""
from __future__ import annotations

import logging
from typing import Callable

from application.ports.model import AnomalyModel
from adapters.outbound.patchcore.model import PatchCore
from adapters.outbound.patchcore.resources import (
    DEFAULT_BUFFER_ZONE_PATH,
    DEFAULT_MEMORY_BANK_PATH,
    DEFAULT_ONNX_PATH,
    load_resources,
)
from config.settings import ModelConfig

logger = logging.getLogger(__name__)


class ModelFactory:
    """Creates anomaly-detection models by name (Factory pattern).

    The factory is not a singleton; the singleton is :class:`PatchCore` itself,
    which the builder instantiates.
    """

    def __init__(self) -> None:
        self._builders: dict[str, Callable[..., AnomalyModel]] = {
            "patchcore": self._build_patchcore,
        }

    def register(self, name: str, builder: Callable[..., AnomalyModel]) -> None:
        """Register a builder for a new model type."""
        self._builders[name.lower()] = builder

    def create(self, name: str = "patchcore", **kwargs) -> AnomalyModel:
        """Build a ready-to-run model by name.

        Raises:
            ValueError: if ``name`` is not registered.
        """
        builder = self._builders.get(name.lower())
        if builder is None:
            raise ValueError(
                f"Unknown model '{name}'. Available: {sorted(self._builders)}"
            )
        logger.info("Creating model '%s'", name)
        return builder(**kwargs)

    @staticmethod
    def _build_patchcore(
        memory_bank_path: str = DEFAULT_MEMORY_BANK_PATH,
        onnx_path: str = DEFAULT_ONNX_PATH,
        buffer_zone_path: str = DEFAULT_BUFFER_ZONE_PATH,
        image_size: int = 224,
        providers: list[str] | None = None,
    ) -> PatchCore:
        resources = load_resources(
            memory_bank_path, onnx_path, buffer_zone_path, image_size, providers
        )
        return PatchCore(
            resources.session, resources.index, resources.transform, resources.buffer_zone
        )


# Module-level convenience: a shared default factory instance.
_default_factory = ModelFactory()


def get_model(name: str = "patchcore", **kwargs) -> AnomalyModel:
    """Create a model using the default :class:`ModelFactory`."""
    return _default_factory.create(name, **kwargs)


def build_patchcore(config: ModelConfig) -> PatchCore:
    """Build the PatchCore detector from typed configuration.

    Convenience wrapper used by the inference composition root so the wiring
    reads from :class:`~config.settings.ModelConfig` rather than hardcoded paths.
    """
    return _default_factory.create(
        "patchcore",
        memory_bank_path=config.memory_bank_path,
        onnx_path=config.onnx_path,
        buffer_zone_path=config.buffer_zone_path,
        image_size=config.image_size,
    )
