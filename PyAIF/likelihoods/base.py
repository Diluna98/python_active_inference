"""Public likelihood interface."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class LikelihoodModel(Protocol):
    """Minimum metadata required by the v0.1 agent constructor."""

    @property
    def obs_dim(self) -> Sequence[int]:
        ...

    @property
    def modality_dependencies(self) -> Sequence[Sequence[int]]:
        ...

    def validate_states(self, states_dim: Sequence[int]) -> None:
        ...
