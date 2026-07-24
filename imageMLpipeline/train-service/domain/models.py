"""Domain entities and value objects for the training service.

Plain, immutable data structures with no dependency on any framework or SDK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class TrainEvent:
    """Signals that a model (re)training run has been requested.

    The raw message ``payload`` is carried as-is; the well-known fields are
    exposed through helpers. The payload shape is::

        {"event": "train", "model": "patchcore", "images": "/{bucket}/{key}"}
    """

    event: str
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def model_type(self) -> str:
        """The model type to retrain (object-key prefix in the models bucket)."""
        return str(self.payload["model"])

    def image_location(self) -> Tuple[str, str]:
        """Parse the ``images`` path ``/{bucket}/{key}`` into ``(bucket, key)``."""
        raw = str(self.payload["images"])
        parts = PurePosixPath(raw).parts
        # Drop the leading "/" root element produced by an absolute path.
        segments = [p for p in parts if p != "/"]
        if len(segments) < 2:
            raise ValueError(f"Malformed image path '{raw}'; expected '/<bucket>/<key>'")
        bucket = segments[0]
        key = "/".join(segments[1:])
        return bucket, key


@dataclass(frozen=True)
class ImageObject:
    """An image that lives in object storage."""

    bucket: str
    key: str
    content_type: str
    size_bytes: int


@dataclass(frozen=True)
class ModelVersion:
    """A registered model version, as recorded in the ``inference_model`` table.

    Locates the model's assets in object storage: they live in ``bucket`` under
    ``{model_type}/{version}/``. ``model_type`` doubles as the object-key prefix
    (e.g. ``patchcore``) and the semver is split into its numeric components so a
    retrain can derive the next ``patch`` version.
    """

    model_id: str
    model_type: str
    bucket: str
    major: int
    minor: int
    patch: int

    @property
    def version(self) -> str:
        """The semver string, e.g. ``1.0.0``."""
        return f"{self.major}.{self.minor}.{self.patch}"
