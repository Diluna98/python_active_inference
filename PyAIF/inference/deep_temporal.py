"""Configuration and implementation for deep temporal state inference."""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from PyAIF.numerics import factor_dot, log_stable_probability, softmax


@dataclass(frozen=True)
class DeepStateInferenceResult:
    """Diagnostics from marginal message passing over all policies."""

    free_energy: np.ndarray
    accuracy: tuple[float, ...]
    complexity: tuple[float, ...]
    iterations: tuple[int, ...]
    converged: tuple[bool, ...]


@dataclass(frozen=True)
class DeepPolicyInferenceResult:
    """Diagnostics from expected-free-energy evaluation over deep policies."""

    expected_free_energy: np.ndarray
    variational_free_energy: np.ndarray
    policy_posterior: np.ndarray
    risk: tuple[float, ...]
    ambiguity: tuple[float, ...]
    information_gain: tuple[float, ...]


def deep_expected_free_energy(
    risk: float,
    ambiguity: float,
    information_gain: float = 0.0,
) -> float:
    """Combine deep-policy value terms using the existing PyAIF convention."""
    return float(-risk - ambiguity + information_gain)


def deep_categorical_policy_ambiguity(
    likelihoods: Any,
    posterior_trajectory: Any,
    start_time: int,
) -> float:
    """Calculate deep-policy ambiguity from categorical likelihoods."""
    ambiguity = 0.0

    for timestep in range(start_time, len(posterior_trajectory)):
        factor_posteriors = list(posterior_trajectory[timestep])
        outcome_entropy = 0.0
        expected_likelihood_entropy = 0.0

        for likelihood in likelihoods:
            expected_outcome = likelihood
            for posterior in factor_posteriors:
                expected_outcome = np.tensordot(
                    expected_outcome,
                    posterior,
                    axes=(1, 0),
                )
            outcome_entropy += -expected_outcome.dot(
                log_stable_probability(expected_outcome)
            )

            likelihood_entropy = -np.sum(
                likelihood * log_stable_probability(likelihood),
                axis=0,
            )
            expected_entropy = likelihood_entropy
            for posterior in factor_posteriors:
                expected_entropy = np.tensordot(
                    expected_entropy,
                    posterior,
                    axes=(0, 0),
                )
            expected_likelihood_entropy += expected_entropy

        ambiguity += outcome_entropy - expected_likelihood_entropy

    return float(ambiguity)


def deep_categorical_policy_risk(
    likelihoods: Any,
    preferences: Any,
    posterior_trajectory: Any,
    start_time: int,
) -> tuple[float, tuple[tuple[np.ndarray, ...], ...]]:
    """Calculate categorical preference value and predicted observations."""
    risk = 0.0
    predictions = []

    for timestep in range(start_time, len(posterior_trajectory)):
        factor_posteriors = list(posterior_trajectory[timestep])
        timestep_predictions = []

        for modality, likelihood in enumerate(likelihoods):
            expected_observation = factor_dot(
                likelihood,
                factor_posteriors,
            )
            timestep_predictions.append(expected_observation)
            risk += expected_observation.dot(
                preferences[modality][:, timestep]
            )

        predictions.append(tuple(timestep_predictions))

    return float(risk), tuple(predictions)


def infer_deep_temporal_states(
    agent: Any,
    time_step: int,
    *,
    convergence_tolerance: float = np.exp(-8),
) -> DeepStateInferenceResult:
    """Update policy-dependent state beliefs using marginal message passing."""
    agent.model_evd = 0
    agent.accuracy_policy = []
    agent.complexity_policy = []
    iterations = []
    convergence = []

    for policy_index, policy in enumerate(agent.policies):
        depolarization = None
        free_energy = None
        completed_iterations = 0
        policy_converged = False

        for message_passing_index in range(agent.number_of_msg_passing):
            previous_free_energy = copy.deepcopy(free_energy)
            agent.F_policy[policy_index] = previous_free_energy
            free_energy = 0
            accuracy = 0
            complexity = 0

            for factor in range(agent.num_factors):
                for tau in range(agent.planning_from, agent.planning_to):
                    tau_reference = tau % agent.temporal_horizon
                    likelihood_message = np.zeros(agent.states_dim[factor])

                    if time_step % agent.temporal_horizon == 0:
                        depolarization = log_stable_probability(
                            agent.D[factor]
                        )
                    else:
                        depolarization = log_stable_probability(
                            agent.policy_dep_posteriors[
                                policy_index,
                                tau_reference,
                                factor,
                            ]
                        )

                    if tau in agent.observations:
                        likelihood_message = (
                            agent.expected_log_likelihood_einsum(
                                agent.observations[tau],
                                factor,
                                policy_index,
                                tau_reference,
                            )
                        )

                    if tau_reference == 0:
                        if time_step < agent.temporal_horizon:
                            forward_message = log_stable_probability(
                                agent.D[factor]
                            )
                        else:
                            forward_message = log_stable_probability(
                                agent.previous_qs_T[factor]
                            )

                        action = policy[tau_reference, :]
                        future_posterior = agent.policy_dep_posteriors[
                            policy_index,
                            tau_reference + 1,
                            factor,
                        ]
                        backward_message = log_stable_probability(
                            agent.transposed_B[factor][
                                :,
                                :,
                                action[factor],
                            ].dot(future_posterior)
                        )
                    elif tau_reference == agent.temporal_horizon - 1:
                        previous_action = policy[tau_reference - 1, :]
                        previous_posterior = agent.policy_dep_posteriors[
                            policy_index,
                            tau_reference - 1,
                            factor,
                        ]
                        forward_message = log_stable_probability(
                            agent.B[factor][
                                :,
                                :,
                                previous_action[factor],
                            ].dot(previous_posterior)
                        )
                        backward_message = np.zeros(agent.D[factor].shape)
                    else:
                        previous_action = policy[tau_reference - 1, :]
                        previous_posterior = agent.policy_dep_posteriors[
                            policy_index,
                            tau_reference - 1,
                            factor,
                        ]
                        forward_message = log_stable_probability(
                            agent.B[factor][
                                :,
                                :,
                                previous_action[factor],
                            ].dot(previous_posterior)
                        )

                        action = policy[tau_reference, :]
                        future_posterior = agent.policy_dep_posteriors[
                            policy_index,
                            tau_reference + 1,
                            factor,
                        ]
                        backward_message = log_stable_probability(
                            agent.transposed_B[factor][
                                :,
                                :,
                                action[factor],
                            ].dot(future_posterior)
                        )

                    prediction_error = (
                        0.5 * (forward_message + backward_message)
                        + likelihood_message
                        - depolarization
                    )
                    depolarization += prediction_error / agent.timeconst

                    posterior = agent.policy_dep_posteriors[
                        policy_index,
                        tau_reference,
                        factor,
                    ]
                    free_energy_term = posterior.dot(
                        -log_stable_probability(posterior)
                        + 0.5 * (forward_message + backward_message)
                        + likelihood_message
                    )
                    free_energy += free_energy_term
                    agent.policy_dep_posteriors[
                        policy_index,
                        tau_reference,
                        factor,
                    ] = softmax(np.array(depolarization))

                    if time_step == tau:
                        accuracy += np.mean(likelihood_message)
                        updated_posterior = agent.policy_dep_posteriors[
                            policy_index,
                            tau_reference,
                            factor,
                        ]
                        complexity += updated_posterior.dot(
                            -log_stable_probability(updated_posterior)
                            + 0.5 * (
                                forward_message + backward_message
                            )
                        ) / agent.states_dim[factor]

            completed_iterations = message_passing_index + 1
            if (
                message_passing_index > 5
                and previous_free_energy is not None
                and abs(free_energy) - abs(previous_free_energy)
                <= convergence_tolerance
            ):
                agent.F_policy[policy_index] = previous_free_energy
                policy_converged = True
                break

        iterations.append(completed_iterations)
        convergence.append(policy_converged)
        agent.accuracy_policy.append(accuracy)
        agent.complexity_policy.append(complexity)

    return DeepStateInferenceResult(
        free_energy=copy.deepcopy(agent.F_policy),
        accuracy=tuple(float(value) for value in agent.accuracy_policy),
        complexity=tuple(
            float(value) for value in agent.complexity_policy
        ),
        iterations=tuple(iterations),
        converged=tuple(convergence),
    )


def infer_deep_temporal_policies(
    agent: Any,
    trial: int,
    time_step: int,
) -> DeepPolicyInferenceResult:
    """Evaluate categorical deep policies and update their posterior."""
    agent.risk = []
    agent.ambiguity = []
    agent.info_gain = []
    policy_risk = []
    policy_ambiguity = []
    policy_information_gain = []

    for policy_index in range(len(agent.policies)):
        information_gain = 0.0
        start_time = time_step % agent.temporal_horizon
        posterior_trajectory = agent.policy_dep_posteriors[policy_index]
        ambiguity = deep_categorical_policy_ambiguity(
            agent.A,
            posterior_trajectory,
            start_time,
        )
        risk, predictions = deep_categorical_policy_risk(
            agent.A,
            agent.C,
            posterior_trajectory,
            start_time,
        )
        for offset, timestep_predictions in enumerate(predictions):
            timestep = start_time + offset
            for modality, prediction in enumerate(timestep_predictions):
                agent.policy_dep_expected_obs[
                    policy_index,
                    timestep,
                ][modality] = prediction

        if agent.learning_D:
            information_gain += agent.calculate_pD_info_gain(policy_index)
        if agent.learning_A:
            information_gain += agent.calculate_pA_info_gain(
                time_step,
                policy_index,
            )
        if agent.learning_B:
            information_gain += agent.calculate_pB_info_gain_vectorized(
                time_step,
                policy_index,
            )

        policy_risk.append(float(risk))
        policy_ambiguity.append(float(ambiguity))
        policy_information_gain.append(float(information_gain))
        agent.G_policy[policy_index] = deep_expected_free_energy(
            risk,
            ambiguity,
            information_gain,
        )

    agent.update_policy_posterior(trial, time_step)

    return DeepPolicyInferenceResult(
        expected_free_energy=copy.deepcopy(agent.G_policy),
        variational_free_energy=copy.deepcopy(agent.F_policy),
        policy_posterior=copy.deepcopy(agent.posterior_pi),
        risk=tuple(policy_risk),
        ambiguity=tuple(policy_ambiguity),
        information_gain=tuple(policy_information_gain),
    )


@dataclass(frozen=True)
class DeepTemporalInference:
    horizon: int
    message_passing_iterations: int = 16
    convergence_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("Deep temporal inference requires horizon >= 2.")
        if self.message_passing_iterations < 1:
            raise ValueError("message_passing_iterations must be positive.")
        if self.convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be positive.")

    def infer_states(
        self,
        agent: Any,
        time_step: int,
    ) -> DeepStateInferenceResult:
        """Run deep temporal state inference with this configuration."""
        return infer_deep_temporal_states(
            agent,
            time_step,
            convergence_tolerance=self.convergence_tolerance,
        )

    def infer_policies(
        self,
        agent: Any,
        trial: int,
        time_step: int,
    ) -> DeepPolicyInferenceResult:
        """Evaluate deep categorical policies and update their posterior."""
        return infer_deep_temporal_policies(agent, trial, time_step)
