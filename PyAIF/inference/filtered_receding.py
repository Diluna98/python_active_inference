"""Current-state filtering followed by policy-specific predictive rollouts."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from PyAIF.numerics import log_stable_probability, softmax

from .deep_temporal import infer_deep_temporal_policies
from .receding import RecedingHorizonInference, require_stage

_MIN_PROBABILITY = 1e-16


@dataclass(frozen=True)
class FilteredStateInferenceResult:
    """Diagnostics from one policy-independent current-state update."""

    posteriors: tuple[np.ndarray, ...]
    variational_free_energy: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class FilteredRecedingHorizonInference(RecedingHorizonInference):
    """Filter the current state once, then roll it forward under each policy.

    This is the conventional robotics estimator/planner split. State inference
    conditions one current belief on the latest observation. Policy inference
    creates deterministic predictive belief trajectories through ``B`` and
    scores them with the existing deep expected-free-energy implementation.
    Unlike :class:`RecedingHorizonInference`, future messages do not influence
    the current posterior.
    """

    def infer_states(self, agent: Any, time_step: int) -> FilteredStateInferenceResult:
        require_stage(agent, "observed")
        result = infer_current_states(
            agent,
            agent.observations[0],
            agent._receding_prior,
            message_passing_iterations=self.message_passing_iterations,
            convergence_tolerance=self.convergence_tolerance,
        )
        agent.filtered_posteriors = [
            np.asarray(posterior, dtype=float).copy() for posterior in result.posteriors
        ]
        # Current evidence is common to all policies and therefore cancels in
        # the policy softmax. Retain it for compatible policy diagnostics.
        for policy_index in range(agent.num_policies):
            agent.F_policy[policy_index] = -result.variational_free_energy
        agent._receding_stage = "states"
        return result

    def infer_policies(self, agent: Any, trial: int, time_step: int):
        require_stage(agent, "states")
        roll_out_policy_states(agent, agent.filtered_posteriors)
        # Relative time zero is the already-observed present and is identical
        # under every policy. Expected free energy therefore starts at the
        # first predicted state after an action.
        result = infer_deep_temporal_policies(
            agent,
            trial,
            1,
            policy_workers=self.policy_workers,
        )
        agent._receding_stage = "policies"
        return result


def _expected_log_likelihood(
    agent: Any,
    observation: Sequence[float],
    factor: int,
    beliefs: Sequence[np.ndarray],
) -> np.ndarray:
    result = np.zeros(agent.states_dim[factor], dtype=float)
    for modality, dependencies in enumerate(agent.mod_dep):
        if factor not in dependencies:
            continue
        if agent.continous_obs:
            likelihood = agent.likelihood.likelihoods(
                observation[modality],
                modality,
            )
        else:
            likelihood = np.take(
                agent.A[modality],
                observation[modality],
                axis=0,
            )
        arguments: list[Any] = [
            log_stable_probability(likelihood),
            list(dependencies),
        ]
        for dependency in dependencies:
            if dependency != factor:
                arguments.extend([beliefs[dependency], [dependency]])
        arguments.append([factor])
        result += np.einsum(*arguments)
    return result


def _variational_free_energy(
    agent: Any,
    observation: Sequence[float],
    priors: Sequence[np.ndarray],
    beliefs: Sequence[np.ndarray],
) -> float:
    value = 0.0
    for factor, posterior in enumerate(beliefs):
        likelihood = _expected_log_likelihood(
            agent,
            observation,
            factor,
            beliefs,
        )
        value += posterior.dot(
            log_stable_probability(posterior)
            - log_stable_probability(priors[factor])
            - likelihood
        )
    return float(value)


def infer_current_states(
    agent: Any,
    observation: Sequence[float],
    prior: Sequence[np.ndarray],
    *,
    message_passing_iterations: int,
    convergence_tolerance: float,
) -> FilteredStateInferenceResult:
    """Infer one factorized current-state posterior without policy rollouts."""

    priors = []
    for factor, values in enumerate(prior):
        normalized = np.asarray(values, dtype=float).copy()
        if (
            normalized.shape != (agent.states_dim[factor],)
            or np.any(~np.isfinite(normalized))
            or np.any(normalized < 0)
            or normalized.sum() <= 0
        ):
            raise ValueError("Current-state priors must be valid distributions.")
        normalized /= normalized.sum()
        priors.append(normalized)

    current = [values.copy() for values in priors]
    previous_free_energy = None
    change = np.inf
    completed_iterations = 0

    for iteration in range(message_passing_iterations):
        base = [posterior.copy() for posterior in current]
        sweep_results = []
        for factor_order in (
            range(agent.num_factors),
            range(agent.num_factors - 1, -1, -1),
        ):
            sweep = [posterior.copy() for posterior in base]
            for factor in factor_order:
                likelihood = _expected_log_likelihood(
                    agent,
                    observation,
                    factor,
                    sweep,
                )
                posterior = softmax(log_stable_probability(priors[factor]) + likelihood)
                posterior = np.clip(posterior, _MIN_PROBABILITY, 1.0)
                sweep[factor] = posterior / posterior.sum()
            sweep_results.append(sweep)

        current = []
        for factor in range(agent.num_factors):
            posterior = 0.5 * (sweep_results[0][factor] + sweep_results[1][factor])
            posterior = np.clip(posterior, _MIN_PROBABILITY, 1.0)
            current.append(posterior / posterior.sum())

        free_energy = _variational_free_energy(
            agent,
            observation,
            priors,
            current,
        )
        completed_iterations = iteration + 1
        if previous_free_energy is not None:
            change = abs(free_energy - previous_free_energy)
            if change < convergence_tolerance:
                break
        previous_free_energy = free_energy

    return FilteredStateInferenceResult(
        posteriors=tuple(posterior.copy() for posterior in current),
        variational_free_energy=float(free_energy),
        iterations=completed_iterations,
        converged=bool(change < convergence_tolerance),
    )


def roll_out_policy_states(
    agent: Any,
    current_posteriors: Sequence[np.ndarray],
) -> None:
    """Populate policy trajectories from a shared current belief through ``B``."""

    for policy_index, policy in enumerate(agent.policies):
        for factor, posterior in enumerate(current_posteriors):
            agent.policy_dep_posteriors[policy_index, 0, factor] = posterior.copy()
        for policy_step, action in enumerate(policy):
            for factor in range(agent.num_factors):
                previous = agent.policy_dep_posteriors[
                    policy_index,
                    policy_step,
                    factor,
                ]
                prediction = agent.B[factor][:, :, action[factor]].dot(previous)
                agent.policy_dep_posteriors[
                    policy_index,
                    policy_step + 1,
                    factor,
                ] = prediction / prediction.sum()
