"""State-transition helpers, including transitions with factor dependencies.

By default every hidden-state factor ``f`` evolves on its own:
``B[f][next_state, previous_state, action]``. With ``B_dependencies`` a factor
may also depend on the previous states of other factors:

.. code-block:: text

    B_dependencies[f] == [f, g]
    B[f].shape == (states_dim[f], states_dim[f], states_dim[g], controls_dim[f])

    P(s_f[t+1] | s_f[t], s_g[t], u_f[t]) = B[f][:, s_f[t], s_g[t], u_f[t]]

Beliefs remain mean-field factorised. Predictions contract the transition
tensor with the beliefs of every dependency. Smoothing (backward) messages use
the transition tensor normalised over its source axes, which reduces exactly
to the column-normalised transpose used by the original single-factor code.
"""

from __future__ import annotations

import string
from typing import Optional, Sequence

import numpy as np

from PyAIF.numerics import log_stable_probability

_LETTERS = string.ascii_letters


def default_dependencies(num_factors: int) -> tuple[tuple[int, ...], ...]:
    """Return the classic one-factor-per-transition structure."""

    return tuple((factor,) for factor in range(num_factors))


def normalize_dependencies(
    dependencies: Optional[Sequence[Sequence[int]]],
    num_factors: int,
) -> tuple[tuple[int, ...], ...]:
    """Validate dependency indices and return them as nested tuples."""

    if dependencies is None:
        return default_dependencies(num_factors)
    if len(dependencies) != num_factors:
        raise ValueError(
            "B_dependencies must contain one dependency list per hidden-state factor."
        )
    normalized = []
    for factor, factor_dependencies in enumerate(dependencies):
        values = tuple(int(value) for value in factor_dependencies)
        if not values:
            raise ValueError(f"B_dependencies[{factor}] must not be empty.")
        if len(set(values)) != len(values):
            raise ValueError(f"B_dependencies[{factor}] contains duplicates.")
        if any(value < 0 or value >= num_factors for value in values):
            raise ValueError(f"B_dependencies[{factor}] contains an invalid factor.")
        normalized.append(values)
    return tuple(normalized)


def has_factor_dependencies(dependencies: Sequence[Sequence[int]]) -> bool:
    """Return whether any factor transition depends on another factor."""

    return any(tuple(values) != (factor,) for factor, values in enumerate(dependencies))


def validate_transition_shapes(
    B: Sequence[np.ndarray],
    states_dim: Sequence[int],
    controls_dim: Sequence[int],
    dependencies: Sequence[Sequence[int]],
) -> None:
    """Check that every ``B[f]`` matches its declared dependencies."""

    for factor, factor_dependencies in enumerate(dependencies):
        expected = (
            int(states_dim[factor]),
            *(int(states_dim[dependency]) for dependency in factor_dependencies),
            int(controls_dim[factor]),
        )
        actual = tuple(np.asarray(B[factor]).shape)
        if actual != expected:
            raise ValueError(
                f"B[{factor}] has shape {actual}; expected {expected} for "
                f"B_dependencies[{factor}] == {list(factor_dependencies)}."
            )


def children_of(
    dependencies: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """For each factor, list the factors whose transitions depend on it."""

    children = [[] for _ in dependencies]
    for child, factor_dependencies in enumerate(dependencies):
        for parent in factor_dependencies:
            children[parent].append(child)
    return tuple(tuple(values) for values in children)


def backward_kernels(
    transitions: Sequence[np.ndarray],
    dependencies: Sequence[Sequence[int]],
) -> tuple[np.ndarray, ...]:
    """Normalise each transition tensor over its source (dependency) axes.

    For a single-factor transition this equals the column-normalised
    transpose ``B[f]^T`` used by the original message-passing code, stored
    with the next-state axis first.
    """

    kernels = []
    for transition, factor_dependencies in zip(transitions, dependencies):
        transition = np.asarray(transition, dtype=float)
        source_axes = tuple(range(1, 1 + len(factor_dependencies)))
        totals = transition.sum(axis=source_axes, keepdims=True)
        kernels.append(
            np.divide(
                transition,
                totals,
                out=np.zeros_like(transition),
                where=totals > 0,
            )
        )
    return tuple(kernels)


def _subscripts(factor_dependencies: Sequence[int]) -> tuple[str, list[str]]:
    next_axis = "Z"
    source_axes = [_LETTERS[index] for index in range(len(factor_dependencies))]
    return next_axis, source_axes


def predict_factor(
    transition: np.ndarray,
    factor_dependencies: Sequence[int],
    posteriors: Sequence[np.ndarray],
    action: int,
) -> np.ndarray:
    """Unnormalised one-step prediction for one factor."""

    tensor = np.asarray(transition)[..., int(action)]
    next_axis, source_axes = _subscripts(factor_dependencies)
    operands = [tensor]
    inputs = [next_axis + "".join(source_axes)]
    for axis, dependency in zip(source_axes, factor_dependencies):
        operands.append(np.asarray(posteriors[dependency], dtype=float))
        inputs.append(axis)
    return np.einsum(",".join(inputs) + "->" + next_axis, *operands)


def predict_states(
    transitions: Sequence[np.ndarray],
    dependencies: Sequence[Sequence[int]],
    posteriors: Sequence[np.ndarray],
    actions: Sequence[int],
) -> list[np.ndarray]:
    """Normalised one-step predictions for every factor."""

    predictions = []
    for factor, factor_dependencies in enumerate(dependencies):
        prediction = predict_factor(
            transitions[factor],
            factor_dependencies,
            posteriors,
            actions[factor],
        )
        predictions.append(prediction / prediction.sum())
    return predictions


def batched_forward_log_message(
    transition: np.ndarray,
    factor_dependencies: Sequence[int],
    previous_posteriors: Sequence[np.ndarray],
    actions: np.ndarray,
) -> np.ndarray:
    """Log forward messages for a batch of policies.

    ``previous_posteriors[d]`` has shape ``(num_policies, states_dim[d])`` and
    ``actions`` holds this factor's action for every policy.
    """

    tensor = np.moveaxis(np.asarray(transition)[..., np.asarray(actions)], -1, 0)
    next_axis, source_axes = _subscripts(factor_dependencies)
    operands = [tensor]
    inputs = ["P" + next_axis + "".join(source_axes)]
    for axis, dependency in zip(source_axes, factor_dependencies):
        operands.append(previous_posteriors[dependency])
        inputs.append("P" + axis)
    prediction = np.einsum(",".join(inputs) + "->P" + next_axis, *operands)
    return log_stable_probability(prediction)


def batched_backward_log_message(
    kernels: Sequence[np.ndarray],
    dependencies: Sequence[Sequence[int]],
    children: Sequence[int],
    target_factor: int,
    current_posteriors: Sequence[np.ndarray],
    future_posteriors: Sequence[np.ndarray],
    actions: np.ndarray,
) -> np.ndarray:
    """Sum of log backward messages from every child transition.

    ``current_posteriors`` and ``future_posteriors`` are per-factor arrays of
    shape ``(num_policies, states_dim)``. ``actions`` has shape
    ``(num_policies, num_factors)`` and holds the action applied between the
    current and the future time point.
    """

    num_policies = actions.shape[0]
    message = np.zeros((num_policies, current_posteriors[target_factor].shape[1]))
    for child in children:
        factor_dependencies = dependencies[child]
        tensor = np.moveaxis(
            np.asarray(kernels[child])[..., np.asarray(actions[:, child])],
            -1,
            0,
        )
        next_axis, source_axes = _subscripts(factor_dependencies)
        operands = [tensor, future_posteriors[child]]
        inputs = ["P" + next_axis + "".join(source_axes), "P" + next_axis]
        output_axis = None
        for axis, dependency in zip(source_axes, factor_dependencies):
            if dependency == target_factor:
                output_axis = axis
                continue
            operands.append(current_posteriors[dependency])
            inputs.append("P" + axis)
        contracted = np.einsum(",".join(inputs) + "->P" + output_axis, *operands)
        message += log_stable_probability(contracted)
    return message


def transition_novelty(
    weights: np.ndarray,
    factor_dependencies: Sequence[int],
    previous_posteriors: Sequence[np.ndarray],
    next_posterior: np.ndarray,
    action: int,
) -> float:
    """Expected information gain about Dirichlet transition parameters.

    ``weights`` is the PyAIF ``wnorm`` tensor (non-positive), so the returned
    novelty is non-negative: ``-sum q(s') W(s', sources) prod q(sources)``.
    """

    expected = predict_factor(
        weights,
        factor_dependencies,
        previous_posteriors,
        action,
    )
    return float(-np.dot(np.asarray(next_posterior, dtype=float), expected))


def joint_transition_evidence(
    states_before: Sequence[np.ndarray],
    state_after: np.ndarray,
    transition_support: np.ndarray,
    *,
    strongest_source_only: bool = True,
) -> np.ndarray:
    """Dirichlet evidence ``q(s') x prod q(sources)`` respecting structural zeros.

    With one source factor this matches ``categorical_transition_evidence``.
    """

    evidence = np.asarray(state_after, dtype=float)
    for posterior in states_before:
        evidence = np.multiply.outer(evidence, np.asarray(posterior, dtype=float))
    evidence = evidence * np.asarray(transition_support, dtype=bool)
    if strongest_source_only:
        strongest = np.unravel_index(np.argmax(evidence), evidence.shape)[1:]
        selected = np.zeros_like(evidence)
        index = (slice(None),) + tuple(strongest)
        selected[index] = evidence[index]
        evidence = selected
    return evidence
