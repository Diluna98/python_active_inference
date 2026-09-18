"""Regression tests for continuous, action-conditioned receding windows."""

import copy

import numpy as np
import pytest

from PyAIF.aif_agent import _stable_argmax

from PyAIF import ActiveInfAgent, RecedingHorizonInference
from test_component_api import continuous_likelihood, make_components


def make_agent(*, continuous=False, horizon=3, workers=1, **kwargs):
    inference = RecedingHorizonInference(
        horizon=horizon, message_passing_iterations=10, policy_workers=workers
    )
    model, likelihood = make_components(inference)
    if continuous:
        likelihood = continuous_likelihood()
    return ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=inference,
        action_selection="deterministic",
        **kwargs,
    ).reset()


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("horizon", [2, 3, 4])
def test_every_observation_gets_full_window_and_action(continuous, horizon):
    agent = make_agent(continuous=continuous, horizon=horizon)
    original_D = copy.deepcopy(agent.D)
    for step in range(2 * horizon + 1):
        observation = [-1.0 if step % 2 == 0 else 1.0] if continuous else [step % 2]
        agent.observe(observation, time_step=step)
        assert list(agent.observations) == [0]
        assert (agent.planning_from, agent.planning_to) == (0, horizon)
        agent.infer_states(0, step)
        agent.infer_policies(0, step)
        assert np.isclose(np.asarray(agent.posterior_pi, float).sum(), 1.0)
        for policy in agent.policy_dep_posteriors:
            for belief in policy[:, 0]:
                assert np.all(np.isfinite(belief))
                assert np.isclose(belief.sum(), 1.0)
        # All predicted time points are filled, including the farthest point.
        for policy in agent.policy_dep_expected_obs:
            for predictions in policy:
                assert np.all(np.isfinite(np.asarray(predictions[0], float)))
        expected = int(agent.policies[_stable_argmax(agent.posterior_pi)][0, 0])
        action = agent.select_action()
        assert action.tolist() == [expected]
        assert agent._current_time == step + 1
        assert np.array_equal(agent.D[0], original_D[0])
        agent.step_time(step)  # Older drivers may still call this.


def test_next_prior_uses_current_belief_and_executed_action():
    agent = make_agent()
    agent.observe([0])
    agent.infer_states()
    agent.infer_policies()
    current = agent.bayesian_mod_avg[0, 0].copy()
    agent.select_action()
    # A controller may veto the selected action; use the actual action.
    agent.observe([1], executed_action=[1])
    expected = agent.B[0][:, :, 1] @ current
    assert np.allclose(agent._receding_prior[0], expected / expected.sum())


def test_initial_window_matches_existing_deep_solver():
    from PyAIF import DeepTemporalInference

    receding = make_agent()
    inference = DeepTemporalInference(horizon=3, message_passing_iterations=10)
    model, likelihood = make_components(inference)
    fixed = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=inference,
        action_selection="deterministic",
    ).reset()
    for agent in (receding, fixed):
        agent.observe([0])
        agent.infer_states()
        agent.infer_policies()
    assert np.allclose(
        np.asarray(receding.F_policy, float), np.asarray(fixed.F_policy, float)
    )
    assert np.allclose(
        np.asarray(receding.G_policy, float), np.asarray(fixed.G_policy, float)
    )
    assert receding.select_action()[0] == fixed.select_action()[0]


@pytest.mark.parametrize("continuous", [False, True])
def test_reset_reproduces_episode_and_parallel_results(continuous):
    def run(agent):
        agent.reset()
        results = []
        for _ in range(5):
            agent.observe([-1.0] if continuous else [0])
            agent.infer_states()
            agent.infer_policies()
            results.append(np.asarray(agent.G_policy, float).copy())
            agent.select_action()
        return results

    agent = make_agent(continuous=continuous)
    baseline = run(agent)
    assert np.allclose(baseline, run(agent))
    assert np.allclose(baseline, run(make_agent(continuous=continuous, workers=2)))


def test_lifecycle_rejects_stale_or_skipped_observations():
    agent = make_agent()
    with pytest.raises(RuntimeError, match="once per decision"):
        agent.infer_states()
    with pytest.raises(ValueError, match="decision index"):
        agent.observe([0], time_step=4)
    agent.observe([0])
    with pytest.raises(RuntimeError, match="once per decision"):
        agent.observe([1])
    with pytest.raises(RuntimeError, match="once per decision"):
        agent.select_action()
    agent.infer_states()
    agent.infer_policies()
    agent.select_action()
    with pytest.raises(RuntimeError, match="once per decision"):
        agent.select_action()
    with pytest.raises(ValueError, match="valid action"):
        agent.observe([0], executed_action=[3])


@pytest.mark.parametrize(
    "flag", ["learning_A", "learning_B", "learning_C", "learning_D", "learning_E"]
)
def test_learning_is_explicitly_rejected(flag):
    with pytest.raises(ValueError, match="parameter learning"):
        make_agent(**{flag: True})


def test_continuous_sampling_uses_absolute_clock(monkeypatch):
    import PyAIF.inference.deep_temporal as deep

    offsets = []
    original = deep.continuous_policy_terms

    def capture(*args, **kwargs):
        offsets.append(kwargs["seed_offset"])
        return original(*args, **kwargs)

    monkeypatch.setattr(deep, "continuous_policy_terms", capture)
    agent = make_agent(continuous=True)
    agent.likelihood.exact_state_limit = 1
    agent.likelihood.policy_samples = 30
    for step in range(2):
        offsets.clear()
        agent.observe([-1.0])
        agent.infer_states()
        agent.infer_policies()
        count = agent.num_policies * agent.temporal_horizon
        assert sorted(offsets) == list(range(step * count, (step + 1) * count))
        agent.select_action()


def test_next_prior_defaults_to_returned_action():
    agent = make_agent()
    agent.observe([0])
    agent.infer_states()
    agent.infer_policies()
    current = agent.bayesian_mod_avg[0, 0].copy()
    action = agent.select_action()
    expected = agent.B[0][:, :, action[0]] @ current
    action[:] = 1 - action  # Caller mutation must not alter the stored action.
    agent.observe([0])
    assert np.allclose(agent._receding_prior[0], expected / expected.sum())


def test_malformed_custom_policy_rejected_before_inference():
    agent = make_agent()
    agent.policies = [np.zeros((1, 1), dtype=int)]
    with pytest.raises(ValueError, match="horizon - 1"):
        agent.reset()
