"""Named expected-free-energy terms shared by every inference mode.

PyAIF scores a policy with the standard decomposition

.. code-block:: text

    G(pi) = - expected_utility              E_q(o|pi)[ln P(o | C)]
            - state_information_gain        I(o; s | pi)       (epistemic value)
            - parameter_information_gain    novelty of A, B, D parameters
            + policy_cost                   optional user-defined cost

Lower expected free energy is better. Internally the agent stores the policy
score ``agent.G_policy = -G(pi)`` because the policy posterior is
``softmax(ln E + gamma * G_policy)``.

Before version 0.5.0 these terms were reported as ``risk`` (holding the
expected utility in categorical modes and its negative in continuous modes),
``ambiguity`` (holding the state information gain) and ``information_gain``
(holding the negative parameter novelty). The old names remain available as
deprecated aliases.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Sequence

import numpy as np

from PyAIF.numerics import factor_dot, log_stable_probability

PolicyCostFunction = Callable[[Any, int, Any], float]


def expected_free_energy(
    expected_utility: float,
    state_information_gain: float,
    parameter_information_gain: float = 0.0,
    policy_cost: float = 0.0,
) -> float:
    """Combine named terms into the expected free energy (lower is better)."""

    return float(
        -expected_utility
        - state_information_gain
        - parameter_information_gain
        + policy_cost
    )


def _modality_beliefs(factor_posteriors, dependencies, modality):
    if dependencies is None:
        return list(factor_posteriors)
    return [factor_posteriors[factor] for factor in dependencies[modality]]


def categorical_state_information_gain(
    likelihoods: Any,
    posterior_trajectory: Any,
    start_time: int,
    modality_dependencies: Optional[Sequence[Sequence[int]]] = None,
) -> float:
    """Epistemic value ``H[Q(o)] - E_Q(s)[H[P(o|s)]]`` summed over time.

    ``modality_dependencies[m]`` lists the factors that ``likelihoods[m]``
    depends on. When omitted every modality depends on every factor.
    """

    information_gain = 0.0
    for timestep in range(start_time, len(posterior_trajectory)):
        factor_posteriors = list(posterior_trajectory[timestep])

        for modality, likelihood in enumerate(likelihoods):
            beliefs = _modality_beliefs(
                factor_posteriors, modality_dependencies, modality
            )
            expected_outcome = factor_dot(likelihood, beliefs)
            outcome_entropy = -expected_outcome.dot(
                log_stable_probability(expected_outcome)
            )
            likelihood_entropy = -np.sum(
                likelihood * log_stable_probability(likelihood),
                axis=0,
            )
            expected_entropy = likelihood_entropy
            for posterior in beliefs:
                expected_entropy = np.tensordot(
                    expected_entropy,
                    posterior,
                    axes=(0, 0),
                )
            information_gain += outcome_entropy - float(expected_entropy)

    return float(information_gain)


def categorical_expected_utility(
    likelihoods: Any,
    log_preferences: Any,
    posterior_trajectory: Any,
    start_time: int,
    modality_dependencies: Optional[Sequence[Sequence[int]]] = None,
) -> tuple[float, tuple[tuple[np.ndarray, ...], ...]]:
    """Expected log preference ``E_Q(o)[ln P(o|C)]`` and predicted outcomes.

    ``log_preferences[m]`` has shape ``(outcomes, horizon)``.
    """

    utility = 0.0
    predictions = []

    for timestep in range(start_time, len(posterior_trajectory)):
        factor_posteriors = list(posterior_trajectory[timestep])
        timestep_predictions = []

        for modality, likelihood in enumerate(likelihoods):
            expected_observation = factor_dot(
                likelihood,
                _modality_beliefs(factor_posteriors, modality_dependencies, modality),
            )
            timestep_predictions.append(expected_observation)
            utility += expected_observation.dot(log_preferences[modality][:, timestep])

        predictions.append(tuple(timestep_predictions))

    return float(utility), tuple(predictions)


def evaluate_policy_cost(
    agent: Any,
    policy_index: int,
    expected_states: Any,
) -> float:
    """Call the agent's optional ``policy_cost_fn`` and validate its result."""

    cost_function: Optional[PolicyCostFunction] = getattr(agent, "policy_cost_fn", None)
    if cost_function is None:
        return 0.0
    cost = float(cost_function(agent, int(policy_index), expected_states))
    if not np.isfinite(cost):
        raise ValueError("policy_cost_fn must return a finite number.")
    return cost


def deep_expected_states(agent: Any, policy_index: int, start_time: int):
    """Per-timestep factor beliefs that a deep policy is scored on."""

    return tuple(
        tuple(
            np.asarray(agent.policy_dep_posteriors[policy_index, timestep, factor])
            for factor in range(agent.num_factors)
        )
        for timestep in range(start_time, agent.temporal_horizon)
    )


def store_policy_terms(
    agent: Any,
    expected_utility: Sequence[float],
    state_information_gain: Sequence[float],
    parameter_information_gain: Sequence[float],
    policy_cost: Sequence[float],
) -> np.ndarray:
    """Write named terms to the agent and return the expected free energy."""

    utility = np.asarray(expected_utility, dtype=float)
    state_gain = np.asarray(state_information_gain, dtype=float)
    parameter_gain = np.asarray(parameter_information_gain, dtype=float)
    cost = np.asarray(policy_cost, dtype=float)
    free_energy = -utility - state_gain - parameter_gain + cost

    agent.expected_utility = utility.tolist()
    agent.state_information_gain = state_gain.tolist()
    agent.parameter_information_gain = parameter_gain.tolist()
    agent.policy_cost = cost.tolist()
    agent.expected_free_energy = free_energy.tolist()
    # Deprecated attribute names kept for existing research scripts.
    agent.risk = (-utility).tolist()
    agent.ambiguity = state_gain.tolist()
    agent.info_gain = parameter_gain.tolist()
    for policy_index, value in enumerate(free_energy):
        agent.G_policy[policy_index] = -value
    return free_energy


class DeprecatedTermNames:
    """Read-only aliases for the pre-0.5.0 diagnostic names."""

    expected_utility: tuple[float, ...]
    state_information_gain: tuple[float, ...]
    parameter_information_gain: tuple[float, ...]

    @property
    def risk(self) -> tuple[float, ...]:
        """Deprecated: expected preference cost, ``-expected_utility``."""

        warnings.warn(
            "'risk' is deprecated; use 'expected_utility' (risk == -expected_utility).",
            DeprecationWarning,
            stacklevel=2,
        )
        return tuple(-value for value in self.expected_utility)

    @property
    def ambiguity(self) -> tuple[float, ...]:
        """Deprecated: this always held the state information gain."""

        warnings.warn(
            "'ambiguity' is deprecated; use 'state_information_gain'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.state_information_gain

    @property
    def information_gain(self) -> tuple[float, ...]:
        """Deprecated: use ``parameter_information_gain``."""

        warnings.warn(
            "'information_gain' is deprecated; use 'parameter_information_gain'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parameter_information_gain
