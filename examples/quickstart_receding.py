"""Run uninterrupted receding-horizon decisions in a two-state environment."""

import numpy as np

from PyAIF import (
    ActiveInfAgent,
    CategoricalLikelihood,
    GenerativeModel,
    RecedingHorizonInference,
)


def objects(*arrays):
    result = np.empty(len(arrays), dtype=object)
    for index, array in enumerate(arrays):
        result[index] = np.asarray(array, dtype=float)
    return result


def main():
    # The hidden state is left/right; action 0 stays and action 1 switches.
    transitions = np.stack((np.eye(2), np.eye(2)[::-1]), axis=2)
    agent = ActiveInfAgent(
        model=GenerativeModel(
            B=objects(transitions),
            D=objects([1, 1]),
            controls_dim=[2],
            controllable_factors=[0],
        ),
        likelihood=CategoricalLikelihood(
            A=objects([[32, 1], [1, 32]]),
            preferences=objects([[1, 1, 1], [8, 8, 8]]),
            modality_dependencies=[[0]],
        ),
        inference=RecedingHorizonInference(
            horizon=3, message_passing_iterations=10
        ),
        action_selection="deterministic",
    ).reset()
    state = 0
    for step in range(8):
        agent.observe([state])
        agent.infer_states()
        agent.infer_policies()
        action = agent.select_action()
        assert action is not None
        state = int(np.argmax(transitions[:, state, action[0]]))
        print(f"step={step} action={action[0]} next_state={state}")


if __name__ == "__main__":
    main()
