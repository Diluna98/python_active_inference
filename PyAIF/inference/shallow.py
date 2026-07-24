"""Configuration and implementation for single-step categorical inference."""

import copy
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from PyAIF.numerics import log_stable_probability, softmax


_MIN_PROBABILITY = 1e-16


@dataclass(frozen=True)
class ShallowStateInferenceResult:
    """Diagnostics from one shallow categorical state-inference update."""

    posteriors: tuple[np.ndarray, ...]
    variational_free_energy: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class ShallowPolicyInferenceResult:
    """Diagnostics from one shallow categorical policy-inference update."""

    expected_free_energy: np.ndarray
    policy_posterior: np.ndarray
    risk: tuple[float, ...]
    ambiguity: tuple[float, ...]
    information_gain: tuple[float, ...]


def infer_shallow_states(
    agent: Any,
    observation: Sequence[int],
    time_step: int,
    *,
    fixed_factor_index: Optional[int] = None,
    convergence_tolerance: float = 1e-4,
) -> ShallowStateInferenceResult:
    """Update an agent's factorized posterior for one observation.

    The forward/backward factor sweeps intentionally preserve the numerical
    behavior of the original ``ActiveInfAgent.infer_states`` implementation.
    """
    agent.observations_cache[time_step % agent.learning_window] = copy.deepcopy(
        observation
    )

    fixed_prior = [
        (agent.posteriors[factor].copy() if time_step > 0 else agent.D[factor].copy())
        for factor in range(agent.num_factors)
    ]
    current_posteriors = [prior.copy() for prior in fixed_prior]

    iteration = 0
    previous_vfe = None
    change_in_vfe = np.inf

    while iteration < agent.num_iterations and change_in_vfe >= convergence_tolerance:
        base_posteriors = [posterior.copy() for posterior in current_posteriors]
        sweep_results = []

        for factor_order in (
            range(agent.num_factors),
            range(agent.num_factors - 1, -1, -1),
        ):
            sweep_posteriors = [posterior.copy() for posterior in base_posteriors]

            for factor in factor_order:
                # The likelihood contraction reads beliefs for other factors.
                agent.posteriors = sweep_posteriors

                if fixed_factor_index is not None and factor == 0:
                    sweep_posteriors[factor] = np.zeros_like(sweep_posteriors[factor])
                    sweep_posteriors[factor][fixed_factor_index] = 1.0
                    continue

                log_likelihood = agent.expected_log_likelihood_einsum(
                    observation,
                    factor,
                )
                weighted = log_stable_probability(fixed_prior[factor]) + log_likelihood
                weighted -= np.max(weighted)

                posterior = softmax(weighted)
                posterior = np.clip(posterior, _MIN_PROBABILITY, 1.0)
                posterior /= posterior.sum()
                sweep_posteriors[factor] = posterior

            sweep_results.append([posterior.copy() for posterior in sweep_posteriors])

        new_posteriors = []
        for factor in range(agent.num_factors):
            posterior = 0.5 * (sweep_results[0][factor] + sweep_results[1][factor])
            posterior = np.clip(posterior, _MIN_PROBABILITY, 1.0)
            posterior /= posterior.sum()
            new_posteriors.append(posterior)

        variational_free_energy = 0.0
        agent.posteriors = new_posteriors

        for factor in range(agent.num_factors):
            posterior = np.clip(
                new_posteriors[factor],
                _MIN_PROBABILITY,
                1.0,
            )
            posterior /= posterior.sum()
            prior = np.clip(
                fixed_prior[factor],
                _MIN_PROBABILITY,
                1.0,
            )
            prior /= prior.sum()
            log_likelihood = agent.expected_log_likelihood_einsum(
                observation,
                factor,
            )
            variational_free_energy += np.dot(
                posterior,
                log_stable_probability(posterior) - log_stable_probability(prior),
            )
            variational_free_energy -= np.dot(
                posterior,
                log_likelihood,
            )

        if previous_vfe is not None:
            change_in_vfe = abs(variational_free_energy - previous_vfe)

        previous_vfe = variational_free_energy
        current_posteriors = [posterior.copy() for posterior in new_posteriors]
        iteration += 1

    agent.posteriors = [posterior.copy() for posterior in current_posteriors]
    agent.posteriors_cache[time_step % agent.learning_window] = copy.deepcopy(
        agent.posteriors
    )

    return ShallowStateInferenceResult(
        posteriors=tuple(posterior.copy() for posterior in agent.posteriors),
        variational_free_energy=float(previous_vfe),
        iterations=iteration,
        converged=bool(change_in_vfe < convergence_tolerance),
    )


def infer_shallow_policies(
    agent: Any,
    time_step: int,
    *,
    policy_precision: float = 240.0,
) -> ShallowPolicyInferenceResult:
    """Score one-step policies and update the agent's policy posterior."""
    agent.risk = []
    agent.ambiguity = []
    agent.info_gain = []

    for policy_index, policy in enumerate(agent.policies):
        information_gain = 0.0
        expected_states = agent.get_expected_states(policy[0])

        ambiguity = agent.calculate_policy_ambiguity(
            0,
            policy_index,
            expected_states,
        )
        risk = agent.calculate_policy_risk(
            0,
            policy_index,
            expected_states,
        )
        agent.ambiguity.append(ambiguity)
        agent.risk.append(risk)

        if agent.learning_D:
            information_gain += agent.calculate_pD_info_gain(policy_index)
        if agent.learning_A:
            likelihood_information_gain = agent.calculate_pA_info_gain(
                time_step,
                policy_index,
                expected_states,
            )
            information_gain += likelihood_information_gain
            agent.info_gain.append(likelihood_information_gain)
        if agent.learning_B:
            information_gain += agent.calculate_pB_info_gain(
                time_step,
                policy_index,
                expected_states,
            )

        # Retained for compatibility; the legacy score currently disables it.
        cost = agent._calculate_cost(policy_index)
        agent.G_policy[policy_index] = -risk + ambiguity + information_gain - cost * 0

    agent.posterior_pi = softmax(
        np.float64(log_stable_probability(agent.E) + policy_precision * agent.G_policy),
        axis=None,
    )

    return ShallowPolicyInferenceResult(
        expected_free_energy=copy.deepcopy(agent.G_policy),
        policy_posterior=copy.deepcopy(agent.posterior_pi),
        risk=tuple(float(value) for value in agent.risk),
        ambiguity=tuple(float(value) for value in agent.ambiguity),
        information_gain=tuple(float(value) for value in agent.info_gain),
    )


@dataclass(frozen=True)
class ShallowInference:
    message_passing_iterations: int = 16
    convergence_tolerance: float = 1e-4
    horizon: int = 1

    def __post_init__(self) -> None:
        if self.message_passing_iterations < 1:
            raise ValueError("message_passing_iterations must be positive.")
        if self.convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be positive.")

    def infer_states(
        self,
        agent: Any,
        observation: Sequence[int],
        time_step: int,
        *,
        fixed_factor_index: Optional[int] = None,
    ) -> ShallowStateInferenceResult:
        """Run shallow categorical state inference with this configuration."""
        return infer_shallow_states(
            agent,
            observation,
            time_step,
            fixed_factor_index=fixed_factor_index,
            convergence_tolerance=self.convergence_tolerance,
        )

    def infer_policies(
        self,
        agent: Any,
        time_step: int,
        *,
        policy_precision: float = 240.0,
    ) -> ShallowPolicyInferenceResult:
        """Run shallow categorical policy inference."""
        return infer_shallow_policies(
            agent,
            time_step,
            policy_precision=policy_precision,
        )
