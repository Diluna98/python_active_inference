"""Tests for transitions whose next state depends on several factors."""

import numpy as np
import pytest

from PyAIF import (
    ActiveInfAgent,
    CategoricalLikelihood,
    DeepTemporalInference,
    FilteredRecedingHorizonInference,
    GenerativeModel,
    RecedingHorizonInference,
    ShallowInference,
)
from PyAIF.numerics import log_stable_probability
from PyAIF.transitions import (
    backward_kernels,
    batched_backward_log_message,
    batched_forward_log_message,
    joint_transition_evidence,
    predict_states,
    transition_novelty,
)
from PyAIF.learning import categorical_transition_evidence
from test_component_api import object_array


def human_model(inference, *, explicit_trivial=False, controllable_human=False):
    """Factor 0: worker type (static). Factor 1: worker's target.

    Type 0 workers move to target 0, type 1 workers move to target 1.
    Only the target is observed, so the type must be inferred from the
    target transitions through ``B_dependencies``.
    """

    worker_type = np.zeros((2, 2, 1))
    worker_type[:, :, 0] = np.eye(2)
    target = np.zeros((2, 2, 2, 1))  # (next target, target, type, action)
    for previous_target in range(2):
        target[:, previous_target, 0, 0] = [0.9, 0.1]
        target[:, previous_target, 1, 0] = [0.1, 0.9]
    B = object_array(worker_type, target)
    dependencies = [[0], [1, 0]]
    if explicit_trivial:
        B = object_array(worker_type, target[:, :, 0, :])
        dependencies = [[0], [1]]
    model = GenerativeModel(
        B=B,
        D=object_array(np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        controls_dim=[1, 1],
        controllable_factors=[0, 1] if controllable_human else [0],
        B_dependencies=dependencies,
    )
    likelihood = CategoricalLikelihood(
        A=object_array(np.array([[0.99, 0.01], [0.01, 0.99]])),
        preferences=object_array(
            np.zeros((2, inference.horizon)) if inference.horizon > 1 else np.zeros(2)
        ),
        modality_dependencies=[[1]],
    )
    return model, likelihood


def test_generative_model_validates_dependency_shapes():
    worker_type = np.ones((2, 2, 1))
    with pytest.raises(ValueError, match="expected"):
        GenerativeModel(
            B=object_array(worker_type, np.ones((2, 2, 1))),
            D=object_array(np.ones(2), np.ones(2)),
            controls_dim=[1, 1],
            controllable_factors=[0],
            B_dependencies=[[0], [1, 0]],
        )
    with pytest.raises(ValueError, match="invalid factor"):
        GenerativeModel(
            B=object_array(worker_type, np.ones((2, 2, 1))),
            D=object_array(np.ones(2), np.ones(2)),
            controls_dim=[1, 1],
            controllable_factors=[0],
            B_dependencies=[[0], [1, 5]],
        )
    with pytest.raises(ValueError, match="one dependency list"):
        GenerativeModel(
            B=object_array(worker_type, np.ones((2, 2, 1))),
            D=object_array(np.ones(2), np.ones(2)),
            controls_dim=[1, 1],
            controllable_factors=[0],
            B_dependencies=[[0]],
        )


def test_default_dependencies_are_one_factor_per_transition():
    model, _ = human_model(DeepTemporalInference(horizon=2), explicit_trivial=True)
    assert model.B_dependencies == ((0,), (1,))
    assert not model.has_factor_dependencies
    dependent, _ = human_model(DeepTemporalInference(horizon=2))
    assert dependent.B_dependencies == ((0,), (1, 0))
    assert dependent.has_factor_dependencies


def test_single_factor_helpers_match_classic_transition_messages():
    rng = np.random.default_rng(3)
    transition = rng.random((3, 3, 2))
    transition /= transition.sum(axis=0, keepdims=True)
    posteriors = [rng.dirichlet(np.ones(3)) for _ in range(4)]
    future = [rng.dirichlet(np.ones(3)) for _ in range(4)]
    actions = np.array([0, 1, 1, 0])

    forward = batched_forward_log_message(
        transition, (0,), [np.stack(posteriors)], actions
    )
    kernels = backward_kernels([transition], ((0,),))
    backward = batched_backward_log_message(
        kernels,
        ((0,),),
        (0,),
        0,
        [np.stack(posteriors)],
        [np.stack(future)],
        actions[:, None],
    )
    for index, action in enumerate(actions):
        expected_forward = log_stable_probability(
            transition[:, :, action] @ posteriors[index]
        )
        transpose = transition[:, :, action].T
        transpose = transpose / transpose.sum(axis=0, keepdims=True)
        expected_backward = log_stable_probability(transpose @ future[index])
        np.testing.assert_allclose(forward[index], expected_forward)
        np.testing.assert_allclose(backward[index], expected_backward)


def test_predict_states_contracts_every_dependency():
    model, _ = human_model(DeepTemporalInference(horizon=2))
    beliefs = [np.array([0.2, 0.8]), np.array([0.6, 0.4])]
    prediction = predict_states(model.B, model.B_dependencies, beliefs, [0, 0])
    expected = np.einsum("nta,t,a->n", model.B[1][..., 0], beliefs[1], beliefs[0])
    np.testing.assert_allclose(prediction[1], expected / expected.sum())
    np.testing.assert_allclose(prediction[0], beliefs[0])


def run_deep(model, likelihood, inference, observations):
    agent = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=inference,
        action_selection="deterministic",
    ).reset()
    for time_step, observation in enumerate(observations):
        agent.observe([observation])
        agent.infer_states()
        agent.infer_policies()
        if time_step < len(observations) - 1:
            agent.select_action()
    return agent


def test_deep_inference_recovers_hidden_factor_from_dependent_transitions():
    inference = DeepTemporalInference(horizon=3, message_passing_iterations=32)
    model, likelihood = human_model(inference)
    agent = run_deep(model, likelihood, inference, [1, 1, 1])
    worker_type = agent.policy_dep_posteriors[0, 2, 0]
    # Repeatedly moving to target 1 is evidence for worker type 1.
    assert worker_type[1] > 0.75

    agent = run_deep(model, likelihood, inference, [0, 0, 0])
    assert agent.policy_dep_posteriors[0, 2, 0][0] > 0.75


def test_without_dependencies_the_hidden_factor_stays_uninformed():
    inference = DeepTemporalInference(horizon=3, message_passing_iterations=32)
    model, likelihood = human_model(inference, explicit_trivial=True)
    agent = run_deep(model, likelihood, inference, [1, 1, 1])
    np.testing.assert_allclose(agent.policy_dep_posteriors[0, 2, 0], [0.5, 0.5])


def test_shallow_expected_states_use_dependencies():
    inference = ShallowInference()
    model, likelihood = human_model(inference)
    agent = ActiveInfAgent(
        model=model, likelihood=likelihood, inference=inference
    ).reset()
    agent.posteriors = [np.array([0.1, 0.9]), np.array([0.5, 0.5])]
    expected = agent.get_expected_states([0, 0])
    manual = predict_states(model.B, model.B_dependencies, agent.posteriors, [0, 0])
    np.testing.assert_allclose(expected[1] / expected[1].sum(), manual[1])
    agent.observe([1])
    agent.infer_states()
    agent.infer_policies()
    assert np.isclose(np.sum(agent.posterior_pi), 1.0)


@pytest.mark.parametrize(
    "inference_type",
    [RecedingHorizonInference, FilteredRecedingHorizonInference],
)
def test_receding_modes_carry_dependent_predictions(inference_type):
    inference = inference_type(horizon=3, message_passing_iterations=16)
    model, likelihood = human_model(inference)
    agent = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=inference,
        action_selection="deterministic",
    ).reset()
    for _ in range(4):
        agent.observe([1])
        agent.infer_states()
        agent.infer_policies()
        agent.select_action()
    prior = agent._receding_prior
    expected = predict_states(
        model.B, model.B_dependencies, agent._receding_posterior, [0, 0]
    )
    agent.observe([1])
    for factor in range(2):
        np.testing.assert_allclose(agent._receding_prior[factor], expected[factor])
    assert prior is not None
    if inference_type is FilteredRecedingHorizonInference:
        agent.infer_states()
        agent.infer_policies()
        for policy_index, policy in enumerate(agent.policies):
            beliefs = [
                agent.policy_dep_posteriors[policy_index, 0, factor]
                for factor in range(2)
            ]
            predicted = predict_states(
                model.B, model.B_dependencies, beliefs, policy[0]
            )
            for factor in range(2):
                np.testing.assert_allclose(
                    agent.policy_dep_posteriors[policy_index, 1, factor],
                    predicted[factor],
                )


def test_joint_transition_evidence_matches_single_factor_evidence():
    before = np.array([0.2, 0.7, 0.1])
    after = np.array([0.6, 0.4])
    support = np.ones((2, 3), dtype=bool)
    np.testing.assert_allclose(
        joint_transition_evidence([before], after, support),
        categorical_transition_evidence(before, after, support),
    )
    joint = joint_transition_evidence(
        [before, np.array([0.3, 0.7])],
        after,
        np.ones((2, 3, 2), dtype=bool),
    )
    assert joint.shape == (2, 3, 2)
    nonzero_sources = {index[1:] for index in zip(*np.nonzero(joint))}
    assert nonzero_sources == {(1, 1)}


def test_deep_learning_updates_dependent_transition_parameters():
    inference = DeepTemporalInference(horizon=3, message_passing_iterations=16)
    model, likelihood = human_model(inference, controllable_human=True)
    agent = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=inference,
        action_selection="deterministic",
        learning_B=True,
    ).reset()
    before = [np.array(values, dtype=float).copy() for values in agent.pB]
    for time_step, observation in enumerate([1, 1, 1]):
        agent.observe([observation])
        agent.infer_states()
        agent.infer_policies()
        agent.select_action()
    agent.learn()
    assert agent.pB[1].shape == before[1].shape
    assert not np.allclose(agent.pB[1], before[1])
    # Structural zeros are never filled.
    assert np.all((before[1] > 0) | (agent.pB[1] == 0))


def test_transition_novelty_pairs_next_state_axis_and_is_non_negative():
    weights = -np.array([[[0.5], [0.0]], [[0.0], [0.1]]])  # (next, previous, action)
    previous = [np.array([1.0, 0.0])]
    next_belief = np.array([0.0, 1.0])
    # Only W[next=1, previous=0] matters for this belief pair.
    assert transition_novelty(weights, (0,), previous, next_belief, 0) == 0.0
    next_belief = np.array([1.0, 0.0])
    assert transition_novelty(weights, (0,), previous, next_belief, 0) == 0.5


def test_legacy_methods_reject_dependencies():
    inference = DeepTemporalInference(horizon=2)
    model, likelihood = human_model(inference)
    agent = ActiveInfAgent(
        model=model, likelihood=likelihood, inference=inference
    ).reset()
    with pytest.raises(NotImplementedError, match="B_dependencies"):
        agent.infer_states_custom(0, 0)
