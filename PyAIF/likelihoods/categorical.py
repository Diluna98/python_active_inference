"""Categorical observation likelihoods backed by an A matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class CategoricalLikelihood:
    """A categorical likelihood and outcome preferences.

    Parameters
    ----------
    A
        One likelihood tensor per observation modality. Axis zero indexes
        observations and the remaining axes index hidden-state factors.
    preferences
        One preference array per modality.
    modality_dependencies
        Hidden-state factors used by each modality. When omitted, dependencies
        are inferred from the state axes present in each likelihood tensor.
    """

    A: np.ndarray
    preferences: np.ndarray
    _modality_dependencies: Optional[Sequence[Sequence[int]]] = None

    def __post_init__(self) -> None:
        if len(self.A) == 0:
            raise ValueError("A must contain at least one observation modality.")
        if len(self.preferences) != len(self.A):
            raise ValueError("preferences must contain one array per modality.")

        if self._modality_dependencies is None:
            dependencies = tuple(
                tuple(range(max(0, np.asarray(modality).ndim - 1)))
                for modality in self.A
            )
        else:
            if len(self._modality_dependencies) != len(self.A):
                raise ValueError(
                    "modality_dependencies must contain one entry per modality."
                )
            dependencies = tuple(
                tuple(int(factor) for factor in modality)
                for modality in self._modality_dependencies
            )

        object.__setattr__(self, "_modality_dependencies", dependencies)

    @property
    def obs_dim(self) -> tuple[int, ...]:
        return tuple(int(np.asarray(modality).shape[0]) for modality in self.A)

    @property
    def modality_dependencies(self) -> tuple[tuple[int, ...], ...]:
        assert self._modality_dependencies is not None
        return tuple(tuple(modality) for modality in self._modality_dependencies)

    def validate_states(self, states_dim: Sequence[int]) -> None:
        states_dim = tuple(int(size) for size in states_dim)
        for modality_idx, (modality, dependencies) in enumerate(
            zip(self.A, self.modality_dependencies)
        ):
            expected = tuple(states_dim[factor] for factor in dependencies)
            actual = tuple(int(size) for size in np.asarray(modality).shape[1:])
            if actual != expected:
                raise ValueError(
                    f"A[{modality_idx}] has state shape {actual}; expected {expected} "
                    f"from modality_dependencies={dependencies}."
                )
