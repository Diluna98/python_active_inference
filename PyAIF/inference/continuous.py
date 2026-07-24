"""Policy-value operations for continuous observations and discrete states."""

from __future__ import annotations

from itertools import product
from string import ascii_lowercase
from typing import Sequence

import numpy as np

from PyAIF.likelihoods import ContinuousLikelihood
from PyAIF.numerics import log_stable_probability


def _integration_weights(grid: np.ndarray) -> np.ndarray:
    weights = np.empty_like(grid, dtype=float)
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return weights


def _state_samples(
    beliefs: Sequence[np.ndarray],
    likelihood: ContinuousLikelihood,
    *,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_beliefs = []
    for factor, belief in enumerate(beliefs):
        normalized = np.asarray(belief, dtype=float)
        if (
            normalized.ndim != 1
            or np.any(~np.isfinite(normalized))
            or np.any(normalized < 0)
            or normalized.sum() <= 0
        ):
            raise ValueError(
                f"State belief {factor} must be a finite, nonnegative "
                "one-dimensional distribution."
            )
        normalized_beliefs.append(normalized / normalized.sum())

    state_count = int(np.prod([len(belief) for belief in normalized_beliefs]))
    if state_count <= likelihood.exact_state_limit:
        samples = np.asarray(
            list(product(*(range(len(belief)) for belief in normalized_beliefs))),
            dtype=int,
        )
        weights = np.ones(len(samples), dtype=float)
        for factor, belief in enumerate(normalized_beliefs):
            weights *= np.asarray(belief, dtype=float)[samples[:, factor]]
        weights /= weights.sum()
        return samples, weights

    seed = None
    if likelihood.random_seed is not None:
        seed = int(likelihood.random_seed) + int(seed_offset)
    rng = np.random.default_rng(seed)
    samples = np.column_stack(
        [
            rng.choice(len(belief), size=likelihood.policy_samples, p=belief)
            for belief in normalized_beliefs
        ]
    )
    return samples, np.full(likelihood.policy_samples, 1.0 / likelihood.policy_samples)


def _conditional_probability_masses(
    likelihood: ContinuousLikelihood,
    modality: int,
    global_samples: np.ndarray,
) -> np.ndarray:
    dependencies = likelihood.modality_dependencies[modality]
    grid = likelihood.get_o_grid(modality)
    if dependencies:
        selected = tuple(global_samples[:, factor] for factor in dependencies)
        state_samples = selected[0] if len(selected) == 1 else selected
        densities = likelihood.likelihoods_grid_vec(grid, modality, state_samples)
    else:
        densities = np.asarray(
            [likelihood.likelihoods(value, modality) for value in grid]
        ).reshape(1, -1)
        densities = np.repeat(densities, len(global_samples), axis=0)
    masses = densities * _integration_weights(grid)[None, :]
    totals = masses.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError(
            f"Continuous modality {modality} has zero density over its grid."
        )
    return masses / totals


def _joint_predictive(
    conditionals: Sequence[np.ndarray],
    state_weights: np.ndarray,
) -> np.ndarray:
    if len(conditionals) + 1 > len(ascii_lowercase):
        raise ValueError("Too many modalities in one joint preference.")
    sample_axis = ascii_lowercase[0]
    outcome_axes = ascii_lowercase[1 : len(conditionals) + 1]
    expression = ",".join([sample_axis] + [sample_axis + axis for axis in outcome_axes])
    expression += "->" + "".join(outcome_axes)
    return np.einsum(expression, state_weights, *conditionals, optimize=True)


def continuous_policy_terms(
    likelihood: ContinuousLikelihood,
    state_beliefs: Sequence[np.ndarray],
    *,
    seed_offset: int = 0,
) -> tuple[float, float, tuple[np.ndarray, ...]]:
    """Return preference cost, state information gain, and predictions.

    Continuous densities are integrated on each configured observation grid.
    Small latent spaces are enumerated exactly; larger spaces use reproducible
    Monte Carlo samples according to ``ContinuousLikelihood`` configuration.
    """

    samples, state_weights = _state_samples(
        state_beliefs,
        likelihood,
        seed_offset=seed_offset,
    )
    conditionals = tuple(
        _conditional_probability_masses(likelihood, modality, samples)
        for modality in range(len(likelihood.observation_grids))
    )
    predictions = tuple(
        np.einsum("s,so->o", state_weights, conditional) for conditional in conditionals
    )

    information_gain = 0.0
    for predictive, conditional in zip(predictions, conditionals):
        predictive_entropy = -predictive.dot(log_stable_probability(predictive))
        conditional_entropies = -np.sum(
            conditional * log_stable_probability(conditional),
            axis=1,
        )
        information_gain += predictive_entropy - state_weights.dot(
            conditional_entropies
        )

    expected_log_preference = 0.0
    joint_modalities = {
        modality
        for key in likelihood.log_preferences
        if isinstance(key, tuple)
        for modality in key
    }
    for key, log_preference in likelihood.log_preferences.items():
        if isinstance(key, int) and key in joint_modalities:
            continue
        modalities = (key,) if isinstance(key, int) else key
        if len(modalities) == 1:
            predictive = predictions[modalities[0]]
        else:
            predictive = _joint_predictive(
                [conditionals[modality] for modality in modalities],
                state_weights,
            )
        expected_log_preference += float(
            np.sum(predictive * np.asarray(log_preference))
        )
    return (
        float(-expected_log_preference),
        float(information_gain),
        predictions,
    )
