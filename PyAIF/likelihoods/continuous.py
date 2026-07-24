"""Reusable likelihood component for scalar continuous observations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np


PreferenceKey = Union[int, tuple[int, ...]]


@dataclass
class ContinuousLikelihood:
    """A continuous observation model over discrete hidden-state factors.

    Parameters
    ----------
    likelihood_fn
        ``likelihood_fn(value, modality)`` must return the conditional density
        over the hidden-state factors listed for that modality.
    observation_grids
        One one-dimensional integration grid per observation modality.
    log_preferences
        Log preferences on those grids. Integer keys describe one modality;
        tuple keys describe a joint preference over multiple modalities.
    modality_dependencies
        Hidden-state factors used by each observation modality.
    grid_likelihood_fn
        Optional vectorized implementation with signature
        ``(grid, modality, state_samples)``. It must return one row per state
        sample and one column per grid value. A correct fallback is supplied.
    learning_fn
        Optional domain learning hook. It receives observations, factorized
        state beliefs, and the learning rate.
    preference_learning_fn
        Optional domain preference-learning hook. It receives factorized state
        beliefs and the learning rate.
    parameter_information_gain_fn
        Optional hook for model-parameter information gain during planning.
    """

    likelihood_fn: Callable[[float, int], np.ndarray]
    observation_grids: Sequence[np.ndarray]
    log_preferences: Mapping[PreferenceKey, np.ndarray]
    modality_dependencies: Sequence[Sequence[int]]
    grid_likelihood_fn: Optional[Callable[[np.ndarray, int, Any], np.ndarray]] = None
    learning_fn: Optional[Callable[[Any, Any, float], Any]] = None
    preference_learning_fn: Optional[Callable[[Any, float], Any]] = None
    parameter_information_gain_fn: Optional[Callable[[Sequence[np.ndarray]], float]] = (
        None
    )
    policy_samples: int = 500
    exact_state_limit: int = 4096
    random_seed: Optional[int] = 0
    model: Any = None

    def __post_init__(self) -> None:
        self.observation_grids = tuple(
            np.asarray(grid, dtype=float) for grid in self.observation_grids
        )
        self.modality_dependencies = tuple(
            tuple(int(factor) for factor in dependencies)
            for dependencies in self.modality_dependencies
        )
        self.log_preferences = {
            self._normalize_preference_key(key): np.asarray(value, dtype=float)
            for key, value in self.log_preferences.items()
        }

        if not self.observation_grids:
            raise ValueError("observation_grids must contain at least one grid.")
        if len(self.observation_grids) != len(self.modality_dependencies):
            raise ValueError(
                "modality_dependencies must contain one entry per modality."
            )
        if any(grid.ndim != 1 or len(grid) < 2 for grid in self.observation_grids):
            raise ValueError(
                "Each continuous observation grid must be one-dimensional "
                "and contain at least two values."
            )
        if any(np.any(np.diff(grid) <= 0) for grid in self.observation_grids):
            raise ValueError("Continuous observation grids must be increasing.")
        if self.policy_samples < 1:
            raise ValueError("policy_samples must be positive.")
        if self.exact_state_limit < 1:
            raise ValueError("exact_state_limit must be positive.")

        for key, preference in self.log_preferences.items():
            modalities = (key,) if isinstance(key, int) else key
            expected = tuple(len(self.observation_grids[index]) for index in modalities)
            if preference.ndim == 0:
                preference = np.full(expected, float(preference))
                self.log_preferences[key] = preference
            if preference.shape != expected:
                raise ValueError(
                    f"log_preferences[{key!r}] has shape {preference.shape}; "
                    f"expected {expected}."
                )

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        modality_dependencies: Sequence[Sequence[int]],
        grid_size: int = 100,
        learning_fn: Optional[Callable[[Any, Any, float], Any]] = None,
        preference_learning_fn: Optional[Callable[[Any, float], Any]] = None,
        parameter_information_gain_fn: Optional[
            Callable[[Sequence[np.ndarray]], float]
        ] = None,
        policy_samples: int = 500,
        exact_state_limit: int = 4096,
        random_seed: Optional[int] = 0,
    ) -> "ContinuousLikelihood":
        """Adapt a domain likelihood object without importing it into PyAIF.

        The object must provide ``likelihoods``, ``get_o_grid``, and
        ``log_preferences``. If available, ``likelihoods_grid_vec`` is used as
        the optimized policy-evaluation callback.
        """

        missing = [
            name
            for name in ("likelihoods", "get_o_grid", "log_preferences")
            if not hasattr(model, name)
        ]
        if missing:
            raise TypeError(
                "Continuous likelihood model is missing: " + ", ".join(missing)
            )
        grids = [
            np.asarray(model.get_o_grid(index, N_grid=grid_size), dtype=float)
            for index in range(len(modality_dependencies))
        ]
        return cls(
            likelihood_fn=model.likelihoods,
            observation_grids=grids,
            log_preferences=model.log_preferences,
            modality_dependencies=modality_dependencies,
            grid_likelihood_fn=getattr(model, "likelihoods_grid_vec", None),
            learning_fn=learning_fn,
            preference_learning_fn=preference_learning_fn,
            parameter_information_gain_fn=parameter_information_gain_fn,
            policy_samples=policy_samples,
            exact_state_limit=exact_state_limit,
            random_seed=random_seed,
            model=model,
        )

    @staticmethod
    def _normalize_preference_key(key: PreferenceKey) -> PreferenceKey:
        if isinstance(key, (int, np.integer)):
            return int(key)
        return tuple(int(modality) for modality in key)

    @property
    def obs_dim(self) -> tuple[int, ...]:
        """Number of integration-grid values for each modality."""

        return tuple(len(grid) for grid in self.observation_grids)

    @property
    def preference_dependencies(self) -> tuple[tuple[int, ...], ...]:
        """Groups of modalities that have joint preferences."""

        return tuple(key for key in self.log_preferences if isinstance(key, tuple))

    def validate_states(self, states_dim: Sequence[int]) -> None:
        states_dim = tuple(int(size) for size in states_dim)
        for modality, dependencies in enumerate(self.modality_dependencies):
            if any(factor < 0 or factor >= len(states_dim) for factor in dependencies):
                raise ValueError(
                    f"Modality {modality} contains an invalid hidden-state factor."
                )
            midpoint = self.observation_grids[modality][
                len(self.observation_grids[modality]) // 2
            ]
            actual = np.asarray(self.likelihoods(midpoint, modality)).shape
            expected = tuple(states_dim[factor] for factor in dependencies)
            if actual != expected:
                raise ValueError(
                    f"Continuous modality {modality} has state shape {actual}; "
                    f"expected {expected} from "
                    f"modality_dependencies={dependencies}."
                )

    def likelihoods(self, observation: float, modality: int) -> np.ndarray:
        """Evaluate ``p(observation | hidden states)`` for one modality."""

        density = np.asarray(
            self.likelihood_fn(float(observation), int(modality)),
            dtype=float,
        )
        if np.any(~np.isfinite(density)) or np.any(density < 0):
            raise ValueError(
                "Continuous likelihood densities must be finite and nonnegative."
            )
        return density

    def get_o_grid(self, modality: int, N_grid: Optional[int] = None) -> np.ndarray:
        """Return the configured grid, optionally interpolated to ``N_grid`` values."""

        grid = self.observation_grids[int(modality)]
        if N_grid is None or N_grid == len(grid):
            return grid.copy()
        if N_grid < 2:
            raise ValueError("N_grid must be at least two.")
        return np.linspace(grid[0], grid[-1], int(N_grid))

    def likelihoods_grid_vec(
        self,
        grid: np.ndarray,
        modality: int,
        state_samples: Any,
    ) -> np.ndarray:
        """Evaluate a grid for a vectorized collection of latent-state samples."""

        grid = np.asarray(grid, dtype=float)
        modality = int(modality)
        if self.grid_likelihood_fn is not None:
            result = np.asarray(
                self.grid_likelihood_fn(grid, modality, state_samples),
                dtype=float,
            )
        else:
            samples = (
                (np.asarray(state_samples, dtype=int),)
                if not isinstance(state_samples, tuple)
                else tuple(np.asarray(sample, dtype=int) for sample in state_samples)
            )
            if not samples:
                raise ValueError("state_samples cannot be empty.")
            sample_count = len(samples[0])
            if any(len(sample) != sample_count for sample in samples):
                raise ValueError(
                    "All latent-state sample arrays must have equal length."
                )
            density_grid = np.stack(
                [self.likelihoods(value, modality) for value in grid],
                axis=-1,
            )
            result = density_grid[(*samples, slice(None))]

        if result.ndim == 1:
            result = result[None, :]
        if result.shape[-1] != len(grid):
            raise ValueError(
                "grid_likelihood_fn must return one column per grid value."
            )
        if np.any(~np.isfinite(result)) or np.any(result < 0):
            raise ValueError(
                "Continuous grid likelihoods must be finite and nonnegative."
            )
        return result

    def update(
        self,
        observations: Any,
        state_beliefs: Any,
        learning_rate: float,
    ) -> bool:
        """Run the optional domain-specific continuous parameter update."""

        if self.learning_fn is None:
            return False
        self.learning_fn(observations, state_beliefs, float(learning_rate))
        return True

    def parameter_information_gain(
        self,
        state_beliefs: Sequence[np.ndarray],
    ) -> float:
        """Return optional information gain about likelihood parameters."""

        if self.parameter_information_gain_fn is None:
            return 0.0
        return float(self.parameter_information_gain_fn(state_beliefs))

    def update_preferences(
        self,
        state_beliefs: Any,
        learning_rate: float,
    ) -> bool:
        """Run the optional domain-specific preference update."""

        if self.preference_learning_fn is None:
            return False
        self.preference_learning_fn(state_beliefs, float(learning_rate))
        return True
