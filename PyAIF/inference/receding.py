"""Full-window temporal replanning using the existing message-passing solver."""

import copy
from dataclasses import dataclass

import numpy as np

from .deep_temporal import (
    DeepTemporalInference,
    infer_deep_temporal_policies,
    infer_deep_temporal_states,
)


@dataclass(frozen=True)
class RecedingHorizonInference(DeepTemporalInference):
    """Replan every observation; execute only the first action of a policy.

    ``horizon`` counts state time points, including the current state. Thus
    horizon 3 evaluates two-action sequences. State estimation uses the existing
    policy-conditioned marginal message-passing equations. Windows use relative
    indices; the public agent clock remains absolute and increases every action.
    Online parameter learning is not yet supported for overlapping windows.
    """

    def infer_states(self, agent, time_step):
        require_stage(agent, "observed")
        # Supply the action-conditioned prior without changing the model's D.
        original_prior = agent.D
        agent.D = copy.deepcopy(agent._receding_prior)
        try:
            result = infer_deep_temporal_states(
                agent,
                0,
                convergence_tolerance=self.convergence_tolerance,
                policy_workers=self.policy_workers,
            )
        finally:
            agent.D = original_prior
        agent._receding_stage = "states"
        return result

    def infer_policies(self, agent, trial, time_step):
        require_stage(agent, "states")
        result = infer_deep_temporal_policies(
            agent, trial, 0, policy_workers=self.policy_workers
        )
        agent._receding_stage = "policies"
        return result


def require_stage(agent, expected):
    if getattr(agent, "_receding_stage", None) != expected:
        raise RuntimeError(
            "Receding-horizon calls must follow reset(), then "
            "observe(), infer_states(), infer_policies(), select_action() "
            "once per decision."
        )


def validate_agent(agent):
    for policy in agent.policies:
        values = np.asarray(policy)
        if values.shape != (agent.temporal_horizon - 1, agent.num_factors):
            raise ValueError(
                "Receding policies must have shape (horizon - 1, num_factors)."
            )
        if not np.all(np.isfinite(values)) or np.any(values != values.astype(int)):
            raise ValueError("Policy actions must be finite integers.")
        for factor, controls in enumerate(agent.controls_dim):
            if np.any(values[:, factor] < 0) or np.any(values[:, factor] >= controls):
                raise ValueError("Policy action is outside controls_dim.")
            if factor not in agent.controlable_states and np.any(values[:, factor]):
                raise ValueError("Uncontrolled factors must use action zero.")
    if len(agent.policies) == 0:
        raise ValueError("At least one receding policy is required.")


def reset_receding(agent):
    agent._receding_prior = copy.deepcopy(agent.D)
    agent._receding_posterior = None
    agent._receding_action = None
    agent._receding_stage = "ready"
    # Precision must not leak between independent episodes.
    agent.gamma_previous = 1.0
    agent.beta_posterior = agent.beta_prior


def observe_receding(agent, observation, time_step, executed_action):
    require_stage(agent, "ready")
    if time_step is not None and int(time_step) != agent._current_time:
        raise ValueError("time_step must match the next receding decision index.")
    action = agent._receding_action
    if executed_action is not None:
        if action is None:
            raise ValueError("executed_action requires a preceding selected action.")
        values = np.asarray(executed_action)
        if (
            values.shape != (agent.num_factors,)
            or not np.all(np.isfinite(values))
            or np.any(values != values.astype(int))
            or np.any(values < 0)
            or np.any(values >= np.asarray(agent.controls_dim))
        ):
            raise ValueError(
                "executed_action must contain one valid action per factor."
            )
        action = values.astype(int)
        for factor in range(agent.num_factors):
            if factor not in agent.controlable_states and action[factor] != 0:
                raise ValueError("Uncontrolled factors must use action zero.")
    if action is not None:
        prior = []
        for factor in range(agent.num_factors):
            predicted = agent.B[factor][:, :, action[factor]].dot(
                agent._receding_posterior[factor]
            )
            prior.append(predicted / predicted.sum())
        agent._receding_prior = prior
    agent.initialize_variables()
    agent.observations = {0: observation.copy()}
    agent._pending_observation = observation.copy()
    agent._receding_stage = "observed"


def select_receding_action(agent):
    require_stage(agent, "policies")
    action, _ = agent.choose_action(agent._current_trial, 0)
    full_action = np.zeros(agent.num_factors, dtype=int)
    for factor in agent.controlable_states:
        full_action[factor] = int(action[factor])
    agent._receding_posterior = [
        np.asarray(agent.bayesian_mod_avg[0, factor], dtype=float).copy()
        for factor in range(agent.num_factors)
    ]
    agent._receding_action = full_action.copy()
    agent._current_time += 1
    agent._receding_stage = "ready"
    return full_action
