import numpy as np
import pytest

from PyAIF import (
    ActiveInfAgent,
    CategoricalLikelihood,
    DeepTemporalInference,
    GenerativeModel,
    ShallowInference,
    deep_expected_free_energy,
)
from PyAIF.inference.deep_temporal import _infer_deep_policy_states


def object_array(*arrays):
    result = np.empty(len(arrays), dtype=object)
    for index, array in enumerate(arrays):
        result[index] = np.asarray(array, dtype=float)
    return result


def transition_model():
    # Action 0 preserves the state; action 1 swaps the two states.
    transitions = np.zeros((2, 2, 2), dtype=float)
    transitions[:, :, 0] = np.array([[16.0, 1.0], [1.0, 16.0]])
    transitions[:, :, 1] = np.array([[1.0, 16.0], [16.0, 1.0]])
    return object_array(transitions)


def make_components(inference):
    A = object_array(np.array([[32.0, 1.0], [1.0, 32.0]]))
    if inference.horizon == 1:
        preferences = object_array(np.zeros(2))
    else:
        preferences = object_array(np.zeros((2, inference.horizon)))

    model = GenerativeModel(
        B=transition_model(),
        D=object_array(np.array([1.0, 1.0])),
        controls_dim=[2],
        controllable_factors=[0],
    )
    likelihood = CategoricalLikelihood(
        A=A,
        preferences=preferences,
        modality_dependencies=[[0]],
    )
    return model, likelihood


def test_component_api_requires_all_components():
    model, _ = make_components(ShallowInference())

    with pytest.raises(ValueError, match="must be provided together"):
        ActiveInfAgent(model=model)


def test_v01_rejects_continuous_observation_mode_clearly():
    with pytest.raises(NotImplementedError, match="planned for v0.2"):
        ActiveInfAgent(
            states_dim=[2],
            obs_dim=[1],
            controls_dim=[1],
            controlable_states=[],
            B=transition_model(),
            D=object_array(np.ones(2)),
            continous_obs=True,
        )


def test_categorical_likelihood_rejects_incompatible_state_shape():
    model, likelihood = make_components(ShallowInference())
    incompatible = GenerativeModel(
        B=object_array(np.ones((3, 3, 1))),
        D=object_array(np.ones(3)),
        controls_dim=[1],
        controllable_factors=[],
    )

    likelihood.validate_states(model.states_dim)
    with pytest.raises(ValueError, match="state shape"):
        likelihood.validate_states(incompatible.states_dim)


def test_component_inputs_remain_unchanged_during_agent_learning():
    inference = ShallowInference(message_passing_iterations=2)
    model, likelihood = make_components(inference)
    original_A = [array.copy() for array in likelihood.A]
    original_B = [array.copy() for array in model.B]
    original_C = [array.copy() for array in likelihood.preferences]
    original_D = [array.copy() for array in model.D]

    agent = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=inference,
        learning_A=True,
        learning_B=True,
        learning_D=True,
        learning_window=2,
        learning_rate=0.5,
        forgeting_rate=1.0,
    )
    agent.reset()
    agent.observations_cache[0, 0] = 0
    agent.observations_cache[1, 0] = 1
    agent.posteriors_cache[0, 0] = np.array([0.8, 0.2])
    agent.posteriors_cache[1, 0] = np.array([0.3, 0.7])
    agent.action_posteriors_cache[0, 0] = 0

    result = agent.perform_learning(trial=0, actual_t=1)

    assert result.likelihood
    assert result.transition
    assert result.initial_state
    assert not np.array_equal(agent.pA[0], original_A[0])
    assert not np.array_equal(agent.pB[0], original_B[0])
    assert not np.array_equal(agent.pD[0], original_D[0])
    for actual, expected in zip(likelihood.A, original_A):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(model.B, original_B):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(likelihood.preferences, original_C):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(model.D, original_D):
        np.testing.assert_array_equal(actual, expected)


def test_discrete_shallow_agent_updates_beliefs_and_selects_action():
    inference = ShallowInference(message_passing_iterations=16)
    model, likelihood = make_components(inference)
    agent = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=inference,
        action_selection="deterministic",
    )

    agent.reset()
    agent.observe([0])
    agent.infer_states()

    assert np.isclose(agent.posteriors[0].sum(), 1.0)
    np.testing.assert_allclose(
        agent.posteriors[0],
        np.array([0.9696969696969697, 0.0303030303030303]),
    )
    assert agent.last_state_inference.iterations == 2
    assert agent.last_state_inference.converged
    assert np.isclose(
        agent.last_state_inference.variational_free_energy,
        np.log(2.0),
    )
    np.testing.assert_allclose(
        agent.last_state_inference.posteriors[0],
        agent.posteriors[0],
    )

    G, F = agent.infer_policies()
    np.testing.assert_allclose(
        np.asarray(G, dtype=float),
        np.array([0.9052857847047793, 0.9052857847047793]),
    )
    assert F is None
    np.testing.assert_allclose(agent.posterior_pi, np.array([0.5, 0.5]))
    np.testing.assert_allclose(
        agent.last_policy_inference.expected_free_energy.astype(float),
        np.asarray(G, dtype=float),
    )
    np.testing.assert_allclose(
        agent.last_policy_inference.policy_posterior,
        agent.posterior_pi,
    )
    np.testing.assert_allclose(
        agent.last_policy_inference.risk,
        np.array([-np.log(2.0), -np.log(2.0)]),
    )
    assert len(agent.last_policy_inference.ambiguity) == 2
    assert agent.last_policy_inference.information_gain == ()
    assert agent.select_action().shape == (1,)


def test_discrete_deep_agent_updates_policy_dependent_beliefs():
    inference = DeepTemporalInference(
        horizon=2,
        message_passing_iterations=8,
    )
    model, likelihood = make_components(inference)
    agent = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=inference,
        action_selection="deterministic",
    )

    agent.reset()
    agent.observe([0])
    agent.infer_states()

    for policy_idx in range(agent.num_policies):
        posterior = agent.policy_dep_posteriors[policy_idx, 0, 0]
        assert np.isclose(posterior.sum(), 1.0)
        assert posterior[0] > posterior[1]

    np.testing.assert_allclose(
        np.asarray(agent.last_state_inference.free_energy, dtype=float),
        np.array([-0.2945894368115574, -0.2945894368115575]),
    )
    np.testing.assert_allclose(
        agent.policy_dep_posteriors[0, 0, 0],
        np.array([0.9820932, 0.0179068]),
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        agent.policy_dep_posteriors[0, 1, 0],
        np.array([0.77883179, 0.22116821]),
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        agent.policy_dep_posteriors[1, 1, 0],
        np.array([0.22116821, 0.77883179]),
        rtol=1e-7,
        atol=1e-7,
    )
    assert agent.last_state_inference.iterations == (7, 7)
    assert agent.last_state_inference.converged == (True, True)

    G, F = agent.infer_policies()
    np.testing.assert_allclose(
        np.asarray(G, dtype=float),
        np.array([-0.9190805547355392, -0.9190805547355392]),
    )
    np.testing.assert_allclose(
        np.asarray(F, dtype=float),
        np.array([-0.2945894368115574, -0.2945894368115575]),
    )
    np.testing.assert_allclose(agent.posterior_pi, np.array([0.5, 0.5]))
    np.testing.assert_allclose(
        agent.last_policy_inference.risk,
        np.array([-1.3862943611198906, -1.3862943611198906]),
    )
    np.testing.assert_allclose(
        agent.last_policy_inference.ambiguity,
        np.array([0.4672138063843514, 0.4672138063843514]),
    )
    assert agent.last_policy_inference.information_gain == (0.0, 0.0)
    np.testing.assert_allclose(
        agent.last_policy_inference.policy_posterior,
        agent.posterior_pi,
    )
    selected_action = agent.select_action()
    assert agent.action_history[0, 0] == selected_action[0]


def test_deep_expected_free_energy_combines_policy_terms():
    assert np.isclose(
        deep_expected_free_energy(
            risk=-1.3862943611198906,
            ambiguity=0.4672138063843514,
            information_gain=0.0,
        ),
        -0.9190805547355392,
    )


@pytest.mark.parametrize(
    "inference",
    [
        ShallowInference(policy_workers=2),
        DeepTemporalInference(horizon=2, policy_workers=2),
    ],
)
def test_parallel_policy_execution_matches_serial(inference):
    serial_inference = (
        ShallowInference()
        if inference.horizon == 1
        else DeepTemporalInference(horizon=2)
    )
    serial_model, serial_likelihood = make_components(serial_inference)
    parallel_model, parallel_likelihood = make_components(inference)
    serial_agent = ActiveInfAgent(
        model=serial_model,
        likelihood=serial_likelihood,
        inference=serial_inference,
        action_selection="deterministic",
    )
    parallel_agent = ActiveInfAgent(
        model=parallel_model,
        likelihood=parallel_likelihood,
        inference=inference,
        action_selection="deterministic",
    )

    for agent in (serial_agent, parallel_agent):
        agent.reset()
        agent.observe([0])
        agent.infer_states()
        agent.infer_policies()

    np.testing.assert_allclose(
        np.asarray(parallel_agent.G_policy, dtype=float),
        np.asarray(serial_agent.G_policy, dtype=float),
    )
    np.testing.assert_allclose(
        parallel_agent.posterior_pi,
        serial_agent.posterior_pi,
    )
    if inference.horizon > 1:
        np.testing.assert_allclose(
            np.asarray(parallel_agent.F_policy, dtype=float),
            np.asarray(serial_agent.F_policy, dtype=float),
        )
        for policy_index in range(serial_agent.num_policies):
            for time_step in range(serial_agent.temporal_horizon):
                for factor in range(serial_agent.num_factors):
                    np.testing.assert_allclose(
                        parallel_agent.policy_dep_posteriors[
                            policy_index,
                            time_step,
                            factor,
                        ],
                        serial_agent.policy_dep_posteriors[
                            policy_index,
                            time_step,
                            factor,
                        ],
                    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ShallowInference(policy_workers=0),
        lambda: DeepTemporalInference(horizon=2, policy_workers=0),
    ],
)
def test_policy_workers_must_be_positive(factory):
    with pytest.raises(ValueError, match="policy_workers must be positive"):
        factory()


def test_batched_deep_inference_matches_policy_reference():
    inference = DeepTemporalInference(
        horizon=3,
        message_passing_iterations=8,
    )
    model, likelihood = make_components(inference)
    agent = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=inference,
        action_selection="deterministic",
    )
    agent.reset()
    agent.observe([0])
    references = [
        _infer_deep_policy_states(
            agent,
            policy_index,
            0,
            inference.convergence_tolerance,
        )
        for policy_index in range(agent.num_policies)
    ]

    agent.infer_states()

    assert agent.num_policies == 4
    for policy_index, reference in enumerate(references):
        np.testing.assert_allclose(
            np.asarray(
                agent.policy_dep_posteriors[policy_index].tolist(),
                dtype=float,
            ),
            np.asarray(reference[0].tolist(), dtype=float),
        )
        assert np.isclose(agent.F_policy[policy_index], reference[1])
        assert agent.last_state_inference.iterations[policy_index] == reference[4]
        assert agent.last_state_inference.converged[policy_index] == reference[5]


def test_shallow_policy_scoring_supports_more_policies_than_states():
    inference = ShallowInference(policy_workers=2)
    transitions = object_array(
        np.repeat(np.eye(2)[:, :, None], 2, axis=2),
        np.repeat(np.eye(2)[:, :, None], 2, axis=2),
    )
    model = GenerativeModel(
        B=transitions,
        D=object_array(np.ones(2), np.ones(2)),
        controls_dim=[2, 2],
        controllable_factors=[0, 1],
    )
    likelihood = CategoricalLikelihood(
        A=object_array(np.full((2, 2, 2), 0.5)),
        preferences=object_array(np.zeros(2)),
        modality_dependencies=[[0, 1]],
    )
    agent = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=inference,
    )
    agent.reset()
    agent.observe([0])
    agent.infer_states()
    expected_free_energy, _ = agent.infer_policies()

    assert agent.num_policies == 4
    assert len(expected_free_energy) == 4
    assert np.all(np.isfinite(np.asarray(expected_free_energy, dtype=float)))
