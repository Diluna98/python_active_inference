"""Reusable data structures for discrete Active Inference models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from PyAIF.transitions import (
    has_factor_dependencies,
    normalize_dependencies,
    validate_transition_shapes,
)


@dataclass(frozen=True)
class GenerativeModel:
    """Domain-independent parts of a factorised generative model.

    Observation likelihoods and preferences intentionally do not live here.
    They are supplied by a likelihood component, allowing the same state
    transition model to be paired with categorical or continuous observations.

    ``B_dependencies`` optionally lets a factor's transition depend on the
    previous states of other factors. ``B_dependencies[f]`` lists those
    factors (normally including ``f`` itself) and ``B[f]`` then has shape
    ``(states_dim[f], *[states_dim[d] for d in B_dependencies[f]],
    controls_dim[f])``. When omitted every factor depends only on itself.
    """

    B: np.ndarray
    D: np.ndarray
    controls_dim: Sequence[int]
    controllable_factors: Sequence[int]
    policies: Optional[Sequence[np.ndarray]] = None
    B_dependencies: Optional[Sequence[Sequence[int]]] = None

    def __post_init__(self) -> None:
        states_dim = tuple(int(np.asarray(prior).shape[0]) for prior in self.D)
        controls_dim = tuple(int(size) for size in self.controls_dim)
        controllable_factors = tuple(
            int(factor) for factor in self.controllable_factors
        )

        if not states_dim:
            raise ValueError("D must contain at least one hidden-state factor.")
        if len(controls_dim) != len(states_dim):
            raise ValueError(
                "controls_dim must contain one entry for every hidden-state factor."
            )
        if any(size < 1 for size in controls_dim):
            raise ValueError("Every control dimension must be at least one.")
        if any(
            factor < 0 or factor >= len(states_dim) for factor in controllable_factors
        ):
            raise ValueError("controllable_factors contains an invalid factor index.")
        if len(self.B) != len(states_dim):
            raise ValueError("B must contain one transition array per state factor.")

        dependencies = normalize_dependencies(self.B_dependencies, len(states_dim))
        if self.B_dependencies is not None:
            validate_transition_shapes(self.B, states_dim, controls_dim, dependencies)

        object.__setattr__(self, "controls_dim", controls_dim)
        object.__setattr__(self, "controllable_factors", controllable_factors)
        object.__setattr__(self, "B_dependencies", dependencies)

    @property
    def states_dim(self) -> tuple[int, ...]:
        """Cardinality of each hidden-state factor."""

        return tuple(int(np.asarray(prior).shape[0]) for prior in self.D)

    @property
    def has_factor_dependencies(self) -> bool:
        """Whether any transition depends on a factor other than its own."""

        return has_factor_dependencies(self.B_dependencies)
