"""Domain entities and value objects for the training service.

Plain, immutable data structures with no dependency on any framework or SDK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class TrainEvent:
    """Signals that a model (re)training run has been requested.

    The raw message ``payload`` is carried as-is for now; concrete fields will be
    introduced once the training workflow is defined.
    """

    event: str
    payload: Dict[str, Any] = field(default_factory=dict)
