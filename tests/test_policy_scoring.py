"""Tests for named expected-free-energy terms and the policy cost hook."""

import numpy as np
import pytest

from PyAIF import (
    ActiveInfAgent,
    CategoricalLikelihood,
    DeepTemporalInference,
    GenerativeModel,
    ShallowInference,
)
from test_component_api import object_array


def flip_model(inference, preferred_outcome=1):
    """One factor, two states; action 0 stays and action 1 flips the state."""

    transition = np.zeros((2, 2, 2))
    transition[:, :, 0] = np.eye(2)
    transition[:, :, 1] = np.fliplr(np.eye(2))
    model = GenerativeModel(
        B=object_array(transition),
        D=object_array(np.array([1.0, 0.0])),
        controls_dim=[2],
        controllable_factors=[0],
    )
    preferences = np.zeros(2)
    preferences[preferred_outcome] = 4.0
    if inference.horizon > 1:
        preferences = np.repeat(preferences[:, None], inference.horizon, axis=1)
    likelihood = CategoricalLikelihood(
        A=object_array(np.eye(2) * 0.98 + 0.01),
        preferences=object_array(preferences),
        modality_dependencies=[[0]],
    )
    return model, likelihood


def decide(inference, **agent_options):
    model, likelihood = flip_model(inference)
    agent = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=inference,
        action_selection="deterministic",
        **agent_options,
    ).reset()
    agent.observe([0])
    agent.infer_states()
    agent.infer_policies()
    return agent, agent.select_action()


@pytest.mark.parametrize(
    "inference",
    [ShallowInference(), DeepTemporalInference(horizon=2)],
)
def test_agent_acts_towards_preferred_outcomes(inference):
    # Regression: shallow categorical inference used to negate the utility.
    agent, action = decide(inference)
    assert int(action[0]) == 1
    result = agent.last_policy_inference
    efe = np.asarray(result.expected_free_energy, dtype=float)
    assert efe[1] < efe[0]
    np.testing.assert_allclose(efe, -np.asarray(agent.G_policy, dtype=float))
    np.testing.assert_allclose(
        efe,
        -np.asarray(result.expected_utility)
        - np.asarray(result.state_information_gain)
        - np.asarray(result.parameter_information_gain)
        + np.asarray(result.policy_cost),
    )


@pytest.mark.parametrize(
    "inference",
    [ShallowInference(), DeepTemporalInference(horizon=2)],
)
def test_policy_cost_shifts_expected_free_energy(inference):
    baseline, _ = decide(inference)
    seen = []

    def cost(agent, policy_index, expected_states):
        seen.append((policy_index, expected_states))
        return 100.0 if policy_index == 1 else 0.0

    agent, action = decide(inference, policy_cost_fn=cost)
    assert int(action[0]) == 0
    np.testing.assert_allclose(
        np.asarray(agent.last_policy_inference.expected_free_energy, dtype=float),
        np.asarray(baseline.last_policy_inference.expected_free_energy, dtype=float)
        + np.array([0.0, 100.0]),
    )
    assert agent.last_policy_inference.policy_cost == (0.0, 100.0)
    assert sorted(index for index, _ in seen) == [0, 1]
    states = dict(seen)[1]
    if isinstance(inference, ShallowInference):
        np.testing.assert_allclose(states[0], [0.0, 1.0], atol=1e-6)
    else:
        assert len(states) == 2  # current and one future time point
        np.testing.assert_allclose(states[1][0], [0.0, 1.0], atol=1e-3)


def test_policy_cost_must_be_callable_and_finite():
    inference = ShallowInference()
    model, likelihood = flip_model(inference)
    with pytest.raises(TypeError, match="callable"):
        ActiveInfAgent(
            model=model,
            likelihood=likelihood,
            inference=inference,
            policy_cost_fn=3.0,
        )
    with pytest.raises(ValueError, match="finite"):
        decide(inference, policy_cost_fn=lambda *_: np.nan)


def test_parameter_information_gain_is_non_negative_and_rewarded():
    inference = DeepTemporalInference(horizon=2)
    agent, _ = decide(inference, learning_A=True, learning_B=True, learning_D=True)
    gains = np.asarray(agent.last_policy_inference.parameter_information_gain)
    assert np.all(gains > 0)


def test_deprecated_names_warn_and_map_to_new_terms():
    agent, _ = decide(ShallowInference())
    result = agent.last_policy_inference
    with pytest.deprecated_call():
        risk = result.risk
    with pytest.deprecated_call():
        ambiguity = result.ambiguity
    with pytest.deprecated_call():
        information_gain = result.information_gain
    np.testing.assert_allclose(risk, -np.asarray(result.expected_utility))
    assert ambiguity == result.state_information_gain
    assert information_gain == result.parameter_information_gain
