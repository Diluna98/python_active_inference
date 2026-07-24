import numpy as np
import pytest

from PyAIF import (
    ActiveInfAgent,
    CategoricalLikelihood,
    DeepTemporalInference,
    GenerativeModel,
    ShallowInference,
)


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
        _modality_dependencies=[[0]],
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
    assert agent.posteriors[0][0] > agent.posteriors[0][1]

    G, F = agent.infer_policies()
    assert G.shape == (2,)
    assert F is None
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

    G, F = agent.infer_policies()
    assert G.shape == (2,)
    assert F.shape == (2,)
    assert np.isclose(agent.posterior_pi.sum(), 1.0)
