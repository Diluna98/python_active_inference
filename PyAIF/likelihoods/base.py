"""Public likelihood interface."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class LikelihoodModel(Protocol):
    """Metadata shared by categorical and continuous likelihood components."""

    @property
    def obs_dim(self) -> Sequence[int]: ...

    @property
    def modality_dependencies(self) -> Sequence[Sequence[int]]: ...

    def validate_states(self, states_dim: Sequence[int]) -> None: ...
