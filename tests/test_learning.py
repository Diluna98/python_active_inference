import numpy as np
import pytest

from PyAIF import (
    ActiveInfAgent,
    CategoricalLikelihood,
    DeepTemporalInference,
    GenerativeModel,
    ShallowInference,
    categorical_observation_evidence,
    categorical_transition_evidence,
    update_dirichlet_parameters,
)


def object_array(*arrays):
    result = np.empty(len(arrays), dtype=object)
    for index, array in enumerate(arrays):
        result[index] = np.asarray(array, dtype=float)
    return result


def learning_agent():
    transition = np.ones((2, 2, 2), dtype=float)
    model = GenerativeModel(
        B=object_array(transition),
        D=object_array(np.ones(2)),
        controls_dim=[2],
        controllable_factors=[0],
    )
    likelihood = CategoricalLikelihood(
        A=object_array(np.ones((2, 2))),
        preferences=object_array(np.zeros((2, 2))),
        _modality_dependencies=[[0]],
    )
    agent = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=DeepTemporalInference(
            horizon=2,
            message_passing_iterations=2,
        ),
        learning_A=True,
        learning_B=True,
        learning_D=True,
        learning_E=True,
        learning_rate=0.5,
        forgeting_rate=1.0,
    )
    agent.reset()
    return agent


def shallow_learning_agent():
    transition = np.ones((2, 2, 2), dtype=float)
    model = GenerativeModel(
        B=object_array(transition),
        D=object_array(np.ones(2)),
        controls_dim=[2],
        controllable_factors=[0],
    )
    likelihood = CategoricalLikelihood(
        A=object_array(np.ones((2, 2))),
        preferences=object_array(np.zeros(2)),
        _modality_dependencies=[[0]],
    )
    agent = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=ShallowInference(message_passing_iterations=2),
        learning_A=True,
        learning_B=True,
        learning_D=True,
        learning_window=2,
        learning_rate=0.5,
        forgeting_rate=1.0,
    )
    agent.reset()
    return agent


def test_dirichlet_update_preserves_structural_zeros():
    updated = update_dirichlet_parameters(
        parameters=np.array([[2.0, 0.0], [1.0, 3.0]]),
        baseline=np.array([[1.0, 0.0], [1.0, 1.0]]),
        evidence=np.array([[0.5, 8.0], [0.25, 0.75]]),
        learning_rate=0.4,
        forgetting_rate=0.5,
    )

    np.testing.assert_allclose(
        updated,
        np.array([[1.7, 0.0], [1.1, 2.3]]),
    )

    with pytest.raises(ValueError, match="identical shapes"):
        update_dirichlet_parameters(
            np.ones(2),
            np.ones(3),
            np.ones(2),
            learning_rate=1.0,
        )


def test_categorical_learning_evidence_builders():
    observation_evidence = categorical_observation_evidence(
        observation=1,
        observation_dimension=2,
        factor_posteriors=[
            np.array([0.75, 0.25]),
            np.array([0.4, 0.6]),
        ],
    )
    assert observation_evidence.shape == (2, 2, 2)
    np.testing.assert_allclose(observation_evidence[0], 0.0)
    np.testing.assert_allclose(
        observation_evidence[1],
        np.array([[0.3, 0.45], [0.1, 0.15]]),
    )

    transition_evidence = categorical_transition_evidence(
        state_before=np.array([0.8, 0.2]),
        state_after=np.array([0.3, 0.7]),
        transition_support=np.ones((2, 2), dtype=bool),
    )
    np.testing.assert_allclose(
        transition_evidence,
        np.array([[0.24, 0.0], [0.56, 0.0]]),
    )


def test_deep_categorical_learning_updates_all_enabled_parameters():
    agent = learning_agent()
    agent.observations = {
        0: np.array([0]),
        1: np.array([1]),
    }
    agent.bayesian_mod_avg[0, 0] = np.array([0.8, 0.2])
    agent.bayesian_mod_avg[1, 0] = np.array([0.3, 0.7])
    agent.action_history[0, 0] = 0
    agent.posterior_pi = np.array([0.6, 0.4])

    result = agent.perform_learning(trial=0)

    assert result.likelihood
    assert result.transition
    assert result.initial_state
    assert result.habit
    assert not result.preference

    np.testing.assert_allclose(
        agent.pA[0],
        np.array([[1.4, 1.1], [1.15, 1.35]]),
    )
    np.testing.assert_allclose(
        agent.pB[0][:, :, 0],
        np.array([[1.12, 1.0], [1.28, 1.0]]),
    )
    np.testing.assert_allclose(agent.pD[0], np.array([1.15, 1.35]))
    np.testing.assert_allclose(agent.pE, np.array([1.3, 1.2]))

    np.testing.assert_allclose(agent.A[0].sum(axis=0), np.ones(2))
    np.testing.assert_allclose(agent.B[0][:, :, 0].sum(axis=0), np.ones(2))
    assert np.isclose(agent.D[0].sum(), 1.0)


def test_shallow_categorical_learning_uses_complete_cache_window():
    agent = shallow_learning_agent()
    agent.observations_cache[0, 0] = 0
    agent.observations_cache[1, 0] = 1
    agent.posteriors_cache[0, 0] = np.array([0.8, 0.2])
    agent.posteriors_cache[1, 0] = np.array([0.3, 0.7])
    agent.action_posteriors_cache[0, 0] = 0

    result = agent.perform_learning(trial=0, actual_t=1)

    assert result.likelihood
    assert result.transition
    assert result.initial_state
    assert not result.habit
    assert not result.preference
    np.testing.assert_allclose(
        agent.pA[0],
        np.array([[1.4, 1.1], [1.15, 1.35]]),
    )
    np.testing.assert_allclose(
        agent.pB[0][:, :, 0],
        np.array([[1.12, 1.0], [1.28, 1.0]]),
    )
    np.testing.assert_allclose(agent.pD[0], np.array([1.4, 1.1]))

    unchanged = agent.perform_learning(trial=0, actual_t=2)
    assert not any(
        (
            unchanged.likelihood,
            unchanged.transition,
            unchanged.initial_state,
            unchanged.habit,
            unchanged.preference,
        )
    )
