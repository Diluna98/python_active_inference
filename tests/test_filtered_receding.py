"""Tests for current-state filtering followed by predictive policy rollout."""

import copy

import numpy as np
import pytest

from PyAIF import (
    ActiveInfAgent,
    CategoricalLikelihood,
    FilteredRecedingHorizonInference,
    GenerativeModel,
    ShallowInference,
    deep_categorical_policy_risk,
)
from test_component_api import continuous_likelihood, make_components, object_array


def make_agent(*, continuous=False, horizon=3, workers=1):
    inference = FilteredRecedingHorizonInference(
        horizon=horizon,
        message_passing_iterations=10,
        policy_workers=workers,
    )
    model, likelihood = make_components(inference)
    if continuous:
        likelihood = continuous_likelihood()
    return ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=inference,
        action_selection="deterministic",
    ).reset()


@pytest.mark.parametrize("continuous", [False, True])
def test_state_stage_only_filters_current_state(continuous):
    agent = make_agent(continuous=continuous)
    observation = [-1.0] if continuous else [0]
    agent.observe(observation)
    before = copy.deepcopy(agent.policy_dep_posteriors)
    agent.infer_states()

    assert agent.last_state_inference.iterations <= 10
    for posterior in agent.last_state_inference.posteriors:
        assert np.isclose(posterior.sum(), 1.0)
        assert np.all(np.isfinite(posterior))
    # No policy trajectory is created during current-state estimation.
    for policy_index in range(agent.num_policies):
        for timestep in range(agent.temporal_horizon):
            for factor in range(agent.num_factors):
                assert np.array_equal(
                    agent.policy_dep_posteriors[policy_index, timestep, factor],
                    before[policy_index, timestep, factor],
                )


def test_categorical_current_filter_matches_shallow_inference():
    filtered = make_agent()
    shallow_inference = ShallowInference(
        message_passing_iterations=10,
        convergence_tolerance=filtered.inference.convergence_tolerance,
    )
    model, likelihood = make_components(shallow_inference)
    shallow = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=shallow_inference,
        action_selection="deterministic",
    ).reset()

    for agent in (filtered, shallow):
        agent.observe([0])
        agent.infer_states()
    assert np.allclose(
        filtered.last_state_inference.posteriors[0],
        shallow.last_state_inference.posteriors[0],
    )


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("horizon", [2, 3, 4])
def test_policy_stage_rolls_out_full_horizon_and_always_acts(continuous, horizon):
    agent = make_agent(continuous=continuous, horizon=horizon)
    for step in range(2 * horizon + 1):
        observation = [-1.0 if step % 2 == 0 else 1.0] if continuous else [step % 2]
        agent.observe(observation, time_step=step)
        agent.infer_states()
        current = [posterior.copy() for posterior in agent.filtered_posteriors]
        agent.infer_policies()

        for policy_index, policy in enumerate(agent.policies):
            for factor in range(agent.num_factors):
                assert np.allclose(
                    agent.policy_dep_posteriors[policy_index, 0, factor],
                    current[factor],
                )
            for policy_step, action in enumerate(policy):
                for factor in range(agent.num_factors):
                    expected = (
                        agent.B[factor][:, :, action[factor]]
                        @ agent.policy_dep_posteriors[policy_index, policy_step, factor]
                    )
                    assert np.allclose(
                        agent.policy_dep_posteriors[
                            policy_index, policy_step + 1, factor
                        ],
                        expected / expected.sum(),
                    )
            if not continuous:
                expected_risk, _ = deep_categorical_policy_risk(
                    agent.A,
                    agent.C,
                    agent.policy_dep_posteriors[policy_index],
                    1,
                )
                assert np.isclose(
                    agent.last_policy_inference.risk[policy_index],
                    expected_risk,
                )
        assert np.isclose(np.asarray(agent.posterior_pi, float).sum(), 1.0)
        assert agent.select_action() is not None


@pytest.mark.parametrize("continuous", [False, True])
def test_filter_result_does_not_depend_on_policy_set(continuous):
    baseline = make_agent(continuous=continuous)
    reduced = make_agent(continuous=continuous)
    reduced.policies = reduced.policies[:1]
    reduced.num_policies = 1
    reduced.reset()
    observation = [-1.0] if continuous else [0]
    for agent in (baseline, reduced):
        agent.observe(observation)
        agent.infer_states()
    for baseline_q, reduced_q in zip(
        baseline.filtered_posteriors,
        reduced.filtered_posteriors,
    ):
        assert np.allclose(baseline_q, reduced_q)


def test_joint_likelihood_updates_multiple_factors_before_policy_rollout():
    horizon = 3
    first_transitions = np.empty((2, 2, 2), dtype=float)
    first_transitions[:, :, 0] = np.eye(2)
    first_transitions[:, :, 1] = np.fliplr(np.eye(2))
    second_transitions = np.eye(3)[:, :, None]
    model = GenerativeModel(
        B=object_array(first_transitions, second_transitions),
        D=object_array(np.ones(2), np.ones(3)),
        controls_dim=[2, 1],
        controllable_factors=[0],
    )
    outcome_zero = np.array(
        [
            [0.95, 0.80, 0.65],
            [0.35, 0.20, 0.05],
        ]
    )
    likelihood = CategoricalLikelihood(
        A=object_array(np.stack((outcome_zero, 1.0 - outcome_zero))),
        preferences=object_array(np.zeros((2, horizon))),
        modality_dependencies=[[0, 1]],
    )
    agent = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=FilteredRecedingHorizonInference(horizon=horizon),
        action_selection="deterministic",
    ).reset()

    agent.observe([0])
    agent.infer_states()

    assert agent.filtered_posteriors[0][0] > agent.filtered_posteriors[0][1]
    assert agent.filtered_posteriors[1][0] > agent.filtered_posteriors[1][2]
    shared_current = [posterior.copy() for posterior in agent.filtered_posteriors]
    agent.infer_policies()
    for policy_index in range(agent.num_policies):
        for factor, posterior in enumerate(shared_current):
            assert np.allclose(
                agent.policy_dep_posteriors[policy_index, 0, factor],
                posterior,
            )


@pytest.mark.parametrize("continuous", [False, True])
def test_parallel_policy_scoring_matches_serial(continuous):
    def run(workers):
        agent = make_agent(continuous=continuous, workers=workers)
        observation = [-1.0] if continuous else [0]
        agent.observe(observation)
        agent.infer_states()
        agent.infer_policies()
        return np.asarray(agent.G_policy, float), np.asarray(agent.posterior_pi, float)

    serial = run(1)
    parallel = run(2)
    assert np.allclose(serial[0], parallel[0])
    assert np.allclose(serial[1], parallel[1])
