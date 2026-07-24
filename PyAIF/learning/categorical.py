"""Domain-independent categorical parameter-learning operations."""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from PyAIF.numerics import one_hot


@dataclass(frozen=True)
class CategoricalLearningResult:
    """Summary of the parameter families updated during one learning call."""

    likelihood: bool = False
    transition: bool = False
    initial_state: bool = False
    habit: bool = False
    preference: bool = False


def update_dirichlet_parameters(
    parameters: np.ndarray,
    baseline: np.ndarray,
    evidence: np.ndarray,
    *,
    learning_rate: float,
    forgetting_rate: float = 1.0,
    support: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply the count update used by the original PyAIF learner."""
    parameters = np.asarray(parameters)
    baseline = np.asarray(baseline)
    evidence = np.asarray(evidence)
    if not (parameters.shape == baseline.shape == evidence.shape):
        raise ValueError(
            "parameters, baseline, and evidence must have identical shapes."
        )

    if support is None:
        support = parameters > 0
    support = np.asarray(support, dtype=bool)
    if support.shape != parameters.shape:
        raise ValueError("support must match the parameter shape.")

    updated = (
        forgetting_rate * (parameters - baseline) + baseline + learning_rate * evidence
    )
    return np.where(support, updated, parameters)


def categorical_observation_evidence(
    observation: int,
    observation_dimension: int,
    factor_posteriors: Sequence[np.ndarray],
) -> np.ndarray:
    """Construct joint observation-state evidence for a likelihood tensor."""
    evidence = one_hot(observation, observation_dimension).astype(float)
    for posterior in factor_posteriors:
        evidence = np.multiply.outer(evidence, posterior)
    return evidence


def categorical_transition_evidence(
    state_before: np.ndarray,
    state_after: np.ndarray,
    transition_support: np.ndarray,
    *,
    strongest_source_column_only: bool = True,
) -> np.ndarray:
    """Construct state-transition evidence while respecting structural zeros."""
    evidence = np.outer(state_after, state_before)
    evidence *= np.asarray(transition_support, dtype=bool)

    if strongest_source_column_only:
        strongest_column = np.unravel_index(
            np.argmax(evidence),
            evidence.shape,
        )[1]
        selected = np.zeros_like(evidence)
        selected[:, strongest_column] = evidence[:, strongest_column]
        evidence = selected

    return evidence


def learn_deep_categorical(agent: Any) -> CategoricalLearningResult:
    """Apply configured categorical learning to one completed deep horizon."""
    learned_preference = False
    learned_likelihood = False
    learned_initial_state = False
    learned_transition = False
    learned_habit = False

    if agent.learning_C:
        for time_step in range(agent.temporal_horizon):
            for modality in range(len(agent.pA)):
                agent.pC[modality][:, time_step] += (
                    agent.learning_rate
                    * agent.disparity_nu[time_step, modality]
                    * agent.expected_obs_chosen[time_step, modality]
                )
        learned_preference = True

    if agent.learning_A:
        for time_step in range(agent.temporal_horizon):
            observation = agent.observations[time_step]
            factor_posteriors = [
                agent.bayesian_mod_avg[time_step, factor]
                for factor in range(agent.num_factors)
            ]
            for modality in range(len(agent.pA)):
                evidence = categorical_observation_evidence(
                    int(observation[modality]),
                    agent.obs_dim[modality],
                    factor_posteriors,
                )
                agent.pA[modality] = update_dirichlet_parameters(
                    agent.pA[modality],
                    agent.pA_0[modality],
                    evidence,
                    learning_rate=agent.learning_rate,
                    forgetting_rate=agent.forgeting_rate,
                )
        agent.A = agent._normalize_colums(agent.pA)
        learned_likelihood = True

    if agent.learning_D:
        final_time = agent.temporal_horizon - 1
        for factor in range(agent.num_factors):
            evidence = agent.bayesian_mod_avg[final_time, factor]
            forgetting_rate = (
                0.0 if factor in agent.controlable_states else agent.forgeting_rate
            )
            agent.pD[factor] = update_dirichlet_parameters(
                agent.pD[factor],
                agent.pD_0[factor],
                evidence,
                learning_rate=agent.learning_rate,
                forgetting_rate=forgetting_rate,
                support=evidence >= 0.01,
            )
        agent.D = agent._normalize_colums(agent.pD)
        learned_initial_state = True

    if agent.learning_B:
        for time_step in range(1, agent.temporal_horizon):
            for factor in agent.controlable_states:
                action = int(agent.action_history[factor, time_step - 1])
                transition_slice = agent.B[factor][:, :, action]
                evidence = categorical_transition_evidence(
                    agent.bayesian_mod_avg[time_step - 1, factor],
                    agent.bayesian_mod_avg[time_step, factor],
                    transition_slice > 0,
                )
                agent.pB[factor][:, :, action] = update_dirichlet_parameters(
                    agent.pB[factor][:, :, action],
                    agent.pB_0[factor][:, :, action],
                    evidence,
                    learning_rate=agent.learning_rate,
                    forgetting_rate=agent.forgeting_rate,
                    support=evidence > 0,
                )
        agent.B = agent._normalize_colums(agent.pB)
        learned_transition = True

    if agent.learning_E:
        agent.pE = update_dirichlet_parameters(
            agent.pE,
            agent.pE_0,
            agent.posterior_pi,
            learning_rate=agent.learning_rate,
            forgetting_rate=agent.forgeting_rate,
            support=np.ones_like(agent.pE, dtype=bool),
        )
        learned_habit = True

    return CategoricalLearningResult(
        likelihood=learned_likelihood,
        transition=learned_transition,
        initial_state=learned_initial_state,
        habit=learned_habit,
        preference=learned_preference,
    )


def learn_shallow_categorical(
    agent: Any,
    actual_time: int,
) -> CategoricalLearningResult:
    """Apply configured learning after a complete shallow evidence window."""
    if actual_time % agent.learning_window != agent.learning_window - 1:
        return CategoricalLearningResult()

    learned_likelihood = False
    learned_transition = False
    learned_initial_state = False

    if agent.learning_A:
        for time_index in range(agent.learning_window):
            factor_posteriors = [
                agent.posteriors_cache[time_index, factor]
                for factor in range(agent.num_factors)
            ]
            for modality in range(len(agent.pA)):
                evidence = categorical_observation_evidence(
                    int(agent.observations_cache[time_index, modality]),
                    agent.obs_dim[modality],
                    factor_posteriors,
                )
                agent.pA[modality] = update_dirichlet_parameters(
                    agent.pA[modality],
                    agent.pA_0[modality],
                    evidence,
                    learning_rate=agent.learning_rate,
                    forgetting_rate=agent.forgeting_rate,
                )
        agent.A = agent._normalize_colums(agent.pA)
        learned_likelihood = True

    if agent.learning_B:
        for time_index in range(1, agent.learning_window):
            for factor in agent.controlable_states:
                action = int(agent.action_posteriors_cache[factor, time_index - 1])
                transition_slice = agent.B[factor][:, :, action]
                evidence = categorical_transition_evidence(
                    agent.posteriors_cache[time_index - 1, factor],
                    agent.posteriors_cache[time_index, factor],
                    transition_slice > 0,
                )
                agent.pB[factor][:, :, action] = update_dirichlet_parameters(
                    agent.pB[factor][:, :, action],
                    agent.pB_0[factor][:, :, action],
                    evidence,
                    learning_rate=agent.learning_rate,
                    forgetting_rate=agent.forgeting_rate,
                    support=evidence > 0,
                )
        agent.B = agent._normalize_colums(agent.pB)
        learned_transition = True

    if agent.learning_D:
        for factor in range(agent.num_factors):
            evidence = agent.posteriors_cache[0, factor]
            agent.pD[factor] = update_dirichlet_parameters(
                agent.pD[factor],
                agent.pD_0[factor],
                evidence,
                learning_rate=agent.learning_rate,
                support=agent.pD[factor] > 0,
            )
        agent.D = agent._normalize_colums(agent.pD)
        learned_initial_state = True

    return CategoricalLearningResult(
        likelihood=learned_likelihood,
        transition=learned_transition,
        initial_state=learned_initial_state,
    )
