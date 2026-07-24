"""Configuration and implementation for deep temporal state inference."""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from PyAIF.inference.base import map_policies
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
    return float(risk + ambiguity - information_gain)


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
            risk += expected_observation.dot(preferences[modality][:, timestep])

        predictions.append(tuple(timestep_predictions))

    return float(risk), tuple(predictions)


def _deep_categorical_policy_terms_batch(
    agent: Any,
    start_time: int,
):
    """Vectorize categorical risk and ambiguity over all deep policies."""
    num_policies = len(agent.policies)
    batch_axis = agent.num_factors
    outcome_axis = agent.num_factors + 1
    risk = np.zeros(num_policies)
    ambiguity = np.zeros(num_policies)
    predictions = []

    for timestep in range(start_time, agent.temporal_horizon):
        timestep_predictions = []
        for modality, likelihood in enumerate(agent.A):
            dependencies = agent.mod_dep[modality]
            arguments = [likelihood, [outcome_axis] + list(dependencies)]
            for factor in dependencies:
                factor_posteriors = np.stack(
                    [
                        agent.policy_dep_posteriors[
                            policy_index,
                            timestep,
                            factor,
                        ]
                        for policy_index in range(num_policies)
                    ]
                )
                arguments.extend([factor_posteriors, [batch_axis, factor]])
            arguments.append([batch_axis, outcome_axis])
            expected_outcome = np.einsum(*arguments)
            timestep_predictions.append(expected_outcome)

            outcome_entropy = -np.sum(
                expected_outcome * log_stable_probability(expected_outcome),
                axis=1,
            )
            likelihood_entropy = -np.sum(
                likelihood * log_stable_probability(likelihood),
                axis=0,
            )
            entropy_arguments = [
                likelihood_entropy,
                list(dependencies),
            ]
            for factor in dependencies:
                entropy_arguments.extend(
                    [
                        np.stack(
                            [
                                agent.policy_dep_posteriors[
                                    policy_index,
                                    timestep,
                                    factor,
                                ]
                                for policy_index in range(num_policies)
                            ]
                        ),
                        [batch_axis, factor],
                    ]
                )
            entropy_arguments.append([batch_axis])
            expected_likelihood_entropy = np.einsum(*entropy_arguments)
            ambiguity += outcome_entropy - expected_likelihood_entropy
            risk += expected_outcome.dot(agent.C[modality][:, timestep])

        predictions.append(tuple(timestep_predictions))

    return risk, ambiguity, tuple(predictions)


def _expected_log_likelihood_for_trajectory(
    agent: Any,
    observation: Any,
    factor: int,
    posterior_trajectory: Any,
    tau_reference: int,
) -> np.ndarray:
    """Contract categorical likelihoods against one policy trajectory."""
    result = np.zeros(agent.states_dim[factor])
    for modality_index, dependencies in enumerate(agent.mod_dep):
        if factor not in dependencies:
            continue

        log_likelihood = log_stable_probability(
            np.take(
                agent.A[modality_index],
                observation[modality_index],
                axis=0,
            )
        )
        arguments = [log_likelihood, list(dependencies)]
        for dependency in dependencies:
            if dependency != factor:
                arguments.extend(
                    [
                        posterior_trajectory[tau_reference, dependency],
                        [dependency],
                    ]
                )
        arguments.append([factor])
        result += np.einsum(*arguments)
    return result


def _infer_deep_policy_states(
    agent: Any,
    policy_index: int,
    time_step: int,
    convergence_tolerance: float,
):
    """Infer one policy without mutating shared agent state."""
    policy = agent.policies[policy_index]
    posterior_trajectory = copy.deepcopy(agent.policy_dep_posteriors[policy_index])
    free_energy = None
    reported_free_energy = None
    completed_iterations = 0
    policy_converged = False

    for message_passing_index in range(agent.number_of_msg_passing):
        previous_free_energy = copy.deepcopy(free_energy)
        reported_free_energy = previous_free_energy
        free_energy = 0
        accuracy = 0
        complexity = 0

        for factor in range(agent.num_factors):
            for tau in range(agent.planning_from, agent.planning_to):
                tau_reference = tau % agent.temporal_horizon
                likelihood_message = np.zeros(agent.states_dim[factor])

                if time_step % agent.temporal_horizon == 0:
                    depolarization = log_stable_probability(agent.D[factor])
                else:
                    depolarization = log_stable_probability(
                        posterior_trajectory[tau_reference, factor]
                    )

                if tau in agent.observations:
                    likelihood_message = _expected_log_likelihood_for_trajectory(
                        agent,
                        agent.observations[tau],
                        factor,
                        posterior_trajectory,
                        tau_reference,
                    )

                if tau_reference == 0:
                    if time_step < agent.temporal_horizon:
                        forward_message = log_stable_probability(agent.D[factor])
                    else:
                        forward_message = log_stable_probability(
                            agent.previous_qs_T[factor]
                        )

                    action = policy[tau_reference, :]
                    future_posterior = posterior_trajectory[
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
                    previous_posterior = posterior_trajectory[
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
                    previous_posterior = posterior_trajectory[
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
                    future_posterior = posterior_trajectory[
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

                posterior = posterior_trajectory[tau_reference, factor]
                free_energy += posterior.dot(
                    -log_stable_probability(posterior)
                    + 0.5 * (forward_message + backward_message)
                    + likelihood_message
                )
                posterior_trajectory[tau_reference, factor] = softmax(
                    np.array(depolarization)
                )

                if time_step == tau:
                    accuracy += np.mean(likelihood_message)
                    updated_posterior = posterior_trajectory[tau_reference, factor]
                    complexity += (
                        updated_posterior.dot(
                            -log_stable_probability(updated_posterior)
                            + 0.5 * (forward_message + backward_message)
                        )
                        / agent.states_dim[factor]
                    )

        completed_iterations = message_passing_index + 1
        if (
            message_passing_index > 5
            and previous_free_energy is not None
            and abs(free_energy) - abs(previous_free_energy) <= convergence_tolerance
        ):
            reported_free_energy = previous_free_energy
            policy_converged = True
            break

    return (
        posterior_trajectory,
        reported_free_energy,
        float(accuracy),
        float(complexity),
        completed_iterations,
        policy_converged,
    )


def _infer_deep_policy_batch(
    agent: Any,
    policy_indices: np.ndarray,
    time_step: int,
    convergence_tolerance: float,
):
    """Vectorize deep state inference over a batch of independent policies."""
    policy_indices = np.asarray(policy_indices, dtype=int)
    num_batch_policies = len(policy_indices)
    policies = np.stack([agent.policies[index] for index in policy_indices])
    trajectories = [
        np.stack(
            [
                [
                    agent.policy_dep_posteriors[index, tau, factor]
                    for tau in range(agent.temporal_horizon)
                ]
                for index in policy_indices
            ]
        )
        for factor in range(agent.num_factors)
    ]

    previous_free_energy = np.full(num_batch_policies, np.nan)
    reported_free_energy = np.full(num_batch_policies, np.nan)
    last_accuracy = np.zeros(num_batch_policies)
    last_complexity = np.zeros(num_batch_policies)
    completed_iterations = np.zeros(num_batch_policies, dtype=int)
    converged = np.zeros(num_batch_policies, dtype=bool)
    active = np.ones(num_batch_policies, dtype=bool)

    def expected_log_likelihood(
        observation: Any,
        target_factor: int,
        tau_reference: int,
    ) -> np.ndarray:
        result = np.zeros((num_batch_policies, agent.states_dim[target_factor]))
        batch_axis = agent.num_factors
        for modality_index, dependencies in enumerate(agent.mod_dep):
            if target_factor not in dependencies:
                continue

            log_likelihood = log_stable_probability(
                np.take(
                    agent.A[modality_index],
                    observation[modality_index],
                    axis=0,
                )
            )
            other_dependencies = [
                dependency for dependency in dependencies if dependency != target_factor
            ]
            if not other_dependencies:
                result += np.broadcast_to(log_likelihood, result.shape)
                continue

            arguments = [log_likelihood, list(dependencies)]
            for dependency in other_dependencies:
                arguments.extend(
                    [
                        trajectories[dependency][:, tau_reference, :],
                        [batch_axis, dependency],
                    ]
                )
            arguments.append([batch_axis, target_factor])
            result += np.einsum(*arguments)
        return result

    for message_passing_index in range(agent.number_of_msg_passing):
        reported_free_energy[active] = previous_free_energy[active]
        free_energy = np.zeros(num_batch_policies)
        accuracy = np.zeros(num_batch_policies)
        complexity = np.zeros(num_batch_policies)

        for factor in range(agent.num_factors):
            transition = np.asarray(agent.B[factor])
            reverse_transition = np.asarray(agent.transposed_B[factor])
            for tau in range(agent.planning_from, agent.planning_to):
                tau_reference = tau % agent.temporal_horizon
                likelihood_message = np.zeros(
                    (num_batch_policies, agent.states_dim[factor])
                )

                if time_step % agent.temporal_horizon == 0:
                    depolarization = np.broadcast_to(
                        log_stable_probability(agent.D[factor]),
                        likelihood_message.shape,
                    ).copy()
                else:
                    depolarization = log_stable_probability(
                        trajectories[factor][:, tau_reference, :]
                    )

                if tau in agent.observations:
                    likelihood_message = expected_log_likelihood(
                        agent.observations[tau],
                        factor,
                        tau_reference,
                    )

                if tau_reference == 0:
                    if time_step < agent.temporal_horizon:
                        forward_message = np.broadcast_to(
                            log_stable_probability(agent.D[factor]),
                            likelihood_message.shape,
                        )
                    else:
                        forward_message = np.broadcast_to(
                            log_stable_probability(agent.previous_qs_T[factor]),
                            likelihood_message.shape,
                        )

                    actions = policies[:, tau_reference, factor]
                    matrices = np.transpose(
                        reverse_transition[:, :, actions],
                        (2, 0, 1),
                    )
                    backward_message = log_stable_probability(
                        np.einsum(
                            "pij,pj->pi",
                            matrices,
                            trajectories[factor][:, tau_reference + 1, :],
                        )
                    )
                elif tau_reference == agent.temporal_horizon - 1:
                    actions = policies[:, tau_reference - 1, factor]
                    matrices = np.transpose(
                        transition[:, :, actions],
                        (2, 0, 1),
                    )
                    forward_message = log_stable_probability(
                        np.einsum(
                            "pij,pj->pi",
                            matrices,
                            trajectories[factor][:, tau_reference - 1, :],
                        )
                    )
                    backward_message = np.zeros_like(forward_message)
                else:
                    previous_actions = policies[:, tau_reference - 1, factor]
                    forward_matrices = np.transpose(
                        transition[:, :, previous_actions],
                        (2, 0, 1),
                    )
                    forward_message = log_stable_probability(
                        np.einsum(
                            "pij,pj->pi",
                            forward_matrices,
                            trajectories[factor][:, tau_reference - 1, :],
                        )
                    )

                    actions = policies[:, tau_reference, factor]
                    backward_matrices = np.transpose(
                        reverse_transition[:, :, actions],
                        (2, 0, 1),
                    )
                    backward_message = log_stable_probability(
                        np.einsum(
                            "pij,pj->pi",
                            backward_matrices,
                            trajectories[factor][:, tau_reference + 1, :],
                        )
                    )

                prediction_error = (
                    0.5 * (forward_message + backward_message)
                    + likelihood_message
                    - depolarization
                )
                depolarization += prediction_error / agent.timeconst

                posterior = trajectories[factor][:, tau_reference, :]
                free_energy += np.sum(
                    posterior
                    * (
                        -log_stable_probability(posterior)
                        + 0.5 * (forward_message + backward_message)
                        + likelihood_message
                    ),
                    axis=1,
                )

                shifted = depolarization - np.max(
                    depolarization,
                    axis=1,
                    keepdims=True,
                )
                updated_posterior = np.exp(shifted)
                updated_posterior /= np.sum(
                    updated_posterior,
                    axis=1,
                    keepdims=True,
                )
                trajectories[factor][:, tau_reference, :] = np.where(
                    active[:, None],
                    updated_posterior,
                    posterior,
                )

                if time_step == tau:
                    accuracy += np.mean(likelihood_message, axis=1)
                    complexity += (
                        np.sum(
                            updated_posterior
                            * (
                                -log_stable_probability(updated_posterior)
                                + 0.5 * (forward_message + backward_message)
                            ),
                            axis=1,
                        )
                        / agent.states_dim[factor]
                    )

        completed_iterations[active] = message_passing_index + 1
        last_accuracy[active] = accuracy[active]
        last_complexity[active] = complexity[active]
        newly_converged = (
            active
            & (message_passing_index > 5)
            & np.isfinite(previous_free_energy)
            & (
                np.abs(free_energy) - np.abs(previous_free_energy)
                <= convergence_tolerance
            )
        )
        converged[newly_converged] = True
        previous_free_energy[active] = free_energy[active]
        active[newly_converged] = False
        if not np.any(active):
            break

    return (
        policy_indices,
        trajectories,
        reported_free_energy,
        last_accuracy,
        last_complexity,
        completed_iterations,
        converged,
    )


def infer_deep_temporal_states(
    agent: Any,
    time_step: int,
    *,
    convergence_tolerance: float = np.exp(-8),
    policy_workers: int = 1,
) -> DeepStateInferenceResult:
    """Update policy-dependent state beliefs using marginal message passing."""
    if len(agent.policies) >= 4:
        num_batches = min(policy_workers, len(agent.policies))
        policy_batches = [
            batch
            for batch in np.array_split(
                np.arange(len(agent.policies)),
                num_batches,
            )
            if len(batch)
        ]
        batch_results = map_policies(
            lambda batch_index: _infer_deep_policy_batch(
                agent,
                policy_batches[batch_index],
                time_step,
                convergence_tolerance,
            ),
            len(policy_batches),
            policy_workers,
        )

        agent.model_evd = 0
        accuracy = np.zeros(len(agent.policies))
        complexity = np.zeros(len(agent.policies))
        iterations = np.zeros(len(agent.policies), dtype=int)
        convergence = np.zeros(len(agent.policies), dtype=bool)
        for result in batch_results:
            indices = result[0]
            trajectories = result[1]
            for local_index, policy_index in enumerate(indices):
                for factor in range(agent.num_factors):
                    for tau in range(agent.temporal_horizon):
                        agent.policy_dep_posteriors[
                            policy_index,
                            tau,
                            factor,
                        ] = trajectories[factor][local_index, tau]
                agent.F_policy[policy_index] = result[2][local_index]
            accuracy[indices] = result[3]
            complexity[indices] = result[4]
            iterations[indices] = result[5]
            convergence[indices] = result[6]

        agent.accuracy_policy = accuracy.tolist()
        agent.complexity_policy = complexity.tolist()
        return DeepStateInferenceResult(
            free_energy=copy.deepcopy(agent.F_policy),
            accuracy=tuple(agent.accuracy_policy),
            complexity=tuple(agent.complexity_policy),
            iterations=tuple(int(value) for value in iterations),
            converged=tuple(bool(value) for value in convergence),
        )

    if policy_workers > 1 and len(agent.policies) > 1:
        results = map_policies(
            lambda policy_index: _infer_deep_policy_states(
                agent,
                policy_index,
                time_step,
                convergence_tolerance,
            ),
            len(agent.policies),
            policy_workers,
        )
        for policy_index, result in enumerate(results):
            agent.policy_dep_posteriors[policy_index] = result[0]
            agent.F_policy[policy_index] = result[1]
        agent.model_evd = 0
        agent.accuracy_policy = [result[2] for result in results]
        agent.complexity_policy = [result[3] for result in results]
        return DeepStateInferenceResult(
            free_energy=copy.deepcopy(agent.F_policy),
            accuracy=tuple(agent.accuracy_policy),
            complexity=tuple(agent.complexity_policy),
            iterations=tuple(result[4] for result in results),
            converged=tuple(result[5] for result in results),
        )

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
                        depolarization = log_stable_probability(agent.D[factor])
                    else:
                        depolarization = log_stable_probability(
                            agent.policy_dep_posteriors[
                                policy_index,
                                tau_reference,
                                factor,
                            ]
                        )

                    if tau in agent.observations:
                        likelihood_message = agent.expected_log_likelihood_einsum(
                            agent.observations[tau],
                            factor,
                            policy_index,
                            tau_reference,
                        )

                    if tau_reference == 0:
                        if time_step < agent.temporal_horizon:
                            forward_message = log_stable_probability(agent.D[factor])
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
                        complexity += (
                            updated_posterior.dot(
                                -log_stable_probability(updated_posterior)
                                + 0.5 * (forward_message + backward_message)
                            )
                            / agent.states_dim[factor]
                        )

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
        complexity=tuple(float(value) for value in agent.complexity_policy),
        iterations=tuple(iterations),
        converged=tuple(convergence),
    )


def infer_deep_temporal_policies(
    agent: Any,
    trial: int,
    time_step: int,
    *,
    policy_workers: int = 1,
) -> DeepPolicyInferenceResult:
    """Evaluate categorical deep policies and update their posterior."""
    if len(agent.policies) >= 4:
        start_time = time_step % agent.temporal_horizon
        policy_risk, policy_ambiguity, predictions = (
            _deep_categorical_policy_terms_batch(agent, start_time)
        )

        def information_gain(policy_index: int) -> float:
            value = 0.0
            if agent.learning_D:
                value += agent.calculate_pD_info_gain(policy_index)
            if agent.learning_A:
                value += agent.calculate_pA_info_gain(
                    time_step,
                    policy_index,
                )
            if agent.learning_B:
                value += agent.calculate_pB_info_gain_vectorized(
                    time_step,
                    policy_index,
                )
            return float(value)

        policy_information_gain = np.asarray(
            map_policies(
                information_gain,
                len(agent.policies),
                policy_workers,
            )
        )
        expected_free_energy = policy_risk + policy_ambiguity - policy_information_gain
        for policy_index in range(len(agent.policies)):
            agent.G_policy[policy_index] = expected_free_energy[policy_index]
        for offset, timestep_predictions in enumerate(predictions):
            timestep = start_time + offset
            for modality, modality_predictions in enumerate(timestep_predictions):
                for policy_index, prediction in enumerate(modality_predictions):
                    agent.policy_dep_expected_obs[
                        policy_index,
                        timestep,
                    ][modality] = prediction

        agent.risk = policy_risk.tolist()
        agent.ambiguity = policy_ambiguity.tolist()
        agent.info_gain = policy_information_gain.tolist()
        agent.update_policy_posterior(trial, time_step)
        return DeepPolicyInferenceResult(
            expected_free_energy=copy.deepcopy(agent.G_policy),
            variational_free_energy=copy.deepcopy(agent.F_policy),
            policy_posterior=copy.deepcopy(agent.posterior_pi),
            risk=tuple(agent.risk),
            ambiguity=tuple(agent.ambiguity),
            information_gain=tuple(agent.info_gain),
        )

    if policy_workers > 1 and len(agent.policies) > 1:

        def evaluate_policy(policy_index: int):
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

            return (
                float(risk),
                float(ambiguity),
                float(information_gain),
                deep_expected_free_energy(risk, ambiguity, information_gain),
                predictions,
            )

        results = map_policies(
            evaluate_policy,
            len(agent.policies),
            policy_workers,
        )
        agent.risk = [result[0] for result in results]
        agent.ambiguity = [result[1] for result in results]
        agent.info_gain = [result[2] for result in results]
        for policy_index, result in enumerate(results):
            agent.G_policy[policy_index] = result[3]
            for offset, timestep_predictions in enumerate(result[4]):
                timestep = time_step % agent.temporal_horizon + offset
                for modality, prediction in enumerate(timestep_predictions):
                    agent.policy_dep_expected_obs[
                        policy_index,
                        timestep,
                    ][modality] = prediction

        agent.update_policy_posterior(trial, time_step)
        return DeepPolicyInferenceResult(
            expected_free_energy=copy.deepcopy(agent.G_policy),
            variational_free_energy=copy.deepcopy(agent.F_policy),
            policy_posterior=copy.deepcopy(agent.posterior_pi),
            risk=tuple(agent.risk),
            ambiguity=tuple(agent.ambiguity),
            information_gain=tuple(agent.info_gain),
        )

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
    policy_workers: int = 1

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("Deep temporal inference requires horizon >= 2.")
        if self.message_passing_iterations < 1:
            raise ValueError("message_passing_iterations must be positive.")
        if self.convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be positive.")
        if self.policy_workers < 1:
            raise ValueError("policy_workers must be positive.")

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
            policy_workers=self.policy_workers,
        )

    def infer_policies(
        self,
        agent: Any,
        trial: int,
        time_step: int,
    ) -> DeepPolicyInferenceResult:
        """Evaluate deep categorical policies and update their posterior."""
        return infer_deep_temporal_policies(
            agent,
            trial,
            time_step,
            policy_workers=self.policy_workers,
        )
