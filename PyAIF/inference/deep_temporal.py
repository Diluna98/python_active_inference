"""Configuration and implementation for deep temporal state inference."""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from PyAIF.numerics import log_stable_probability, softmax


@dataclass(frozen=True)
class DeepStateInferenceResult:
    """Diagnostics from marginal message passing over all policies."""

    free_energy: np.ndarray
    accuracy: tuple[float, ...]
    complexity: tuple[float, ...]
    iterations: tuple[int, ...]
    converged: tuple[bool, ...]


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
