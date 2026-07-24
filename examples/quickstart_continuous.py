"""Minimal continuous-observation PyAIF agent."""

import numpy as np

from PyAIF import (
    ActiveInfAgent,
    ContinuousLikelihood,
    GenerativeModel,
    ShallowInference,
)


def object_array(*values):
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = np.asarray(value, dtype=float)
    return result


def build_agent():
    observation_grid = np.linspace(-2.0, 2.0, 101)
    state_means = np.array([-1.0, 1.0])
    observation_sigma = 0.2

    def gaussian_density(observation, modality):
        del modality
        standardized = (observation - state_means) / observation_sigma
        return np.exp(-0.5 * standardized**2) / (
            observation_sigma * np.sqrt(2.0 * np.pi)
        )

    preferred_outcomes = np.exp(2.0 * observation_grid)
    preferred_outcomes /= preferred_outcomes.sum()
    likelihood = ContinuousLikelihood(
        likelihood_fn=gaussian_density,
        observation_grids=[observation_grid],
        log_preferences={0: np.log(preferred_outcomes)},
        modality_dependencies=[[0]],
    )

    transition = np.zeros((2, 2, 2))
    transition[:, :, 0] = np.eye(2)
    transition[:, :, 1] = np.fliplr(np.eye(2))
    model = GenerativeModel(
        B=object_array(transition),
        D=object_array(np.array([0.5, 0.5])),
        controls_dim=[2],
        controllable_factors=[0],
    )
    return ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=ShallowInference(),
        action_selection="deterministic",
    )


def main():
    agent = build_agent()
    agent.reset()
    agent.observe([-1.0])
    agent.infer_states()
    expected_free_energy, _ = agent.infer_policies()
    action = agent.select_action()

    print("state posterior:", agent.posteriors[0])
    print("expected free energy:", np.asarray(expected_free_energy, dtype=float))
    print("selected action:", action)


if __name__ == "__main__":
    main()
