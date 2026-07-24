"""Minimal categorical PyAIF agent using the supported component API."""

import numpy as np

from PyAIF import (
    ActiveInfAgent,
    CategoricalLikelihood,
    GenerativeModel,
    ShallowInference,
)


def object_array(*values):
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = np.asarray(value, dtype=float)
    return result


def build_agent():
    likelihood = object_array(
        np.array(
            [
                [0.95, 0.05],
                [0.05, 0.95],
            ]
        )
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
    observation_model = CategoricalLikelihood(
        A=likelihood,
        preferences=object_array(np.zeros(2)),
        _modality_dependencies=[[0]],
    )
    return ActiveInfAgent(
        model=model,
        likelihood=observation_model,
        inference=ShallowInference(),
        action_selection="deterministic",
    )


def main():
    agent = build_agent()
    agent.reset()
    agent.observe([0])
    agent.infer_states()
    expected_free_energy, _ = agent.infer_policies()
    action = agent.select_action()

    print("state posterior:", agent.posteriors[0])
    print("expected free energy:", np.asarray(expected_free_energy, dtype=float))
    print("selected action:", action)


if __name__ == "__main__":
    main()
