import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
HANDOVER_DIR = REPO_ROOT / "examples" / "handover_task_single_agent"


def load_handover_main():
    module_name = "pyaif_handover_example"
    if module_name in sys.modules:
        return sys.modules[module_name]

    sys.path.insert(0, str(HANDOVER_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            HANDOVER_DIR / "main.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(HANDOVER_DIR))


def test_handover_model_and_policy_fixture():
    handover = load_handover_main()
    agent, horizon = handover.build_agent(
        trials=1,
        message_passing_iterations=2,
    )

    assert horizon == 4
    assert agent.states_dim == [2, 2, 2, 2, 4, 4, 4]
    assert agent.obs_dim == [3, 4, 4, 4, 5, 3, 4]
    assert agent.controls_dim == [1, 1, 1, 1, 4, 1, 1]
    assert agent.controlable_states == [4]
    assert agent.num_policies == 21

    for policy in agent.policies:
        control_sequence = policy[:, 4]
        assert control_sequence[0] != 3
        slot_actions = control_sequence[control_sequence != 3]
        assert len(slot_actions) == len(np.unique(slot_actions))


def test_handover_one_step_numerical_regression():
    handover = load_handover_main()
    agent, _ = handover.build_agent(
        trials=1,
        message_passing_iterations=2,
    )

    # Deterministic observation:
    # safe object, empty slots, command slot1, positive feedback,
    # end effector and command memory at "ideal".
    observation = np.array([0, 3, 3, 3, 0, 0, 3])

    agent.reset()
    agent.observe(observation, time_step=0)
    agent.infer_states_custom(0, 0)
    G, F = agent.infer_policies(0, 0)

    np.testing.assert_allclose(
        np.asarray(F, dtype=float)[[0, 1, 4, 6]],
        [-145.53388023, -136.72470235, -132.66484196, -123.85566407],
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(G, dtype=float)[[0, 6, 13, 20]],
        [41.29793511, 42.50113887, 42.64190891, 42.65249464],
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        agent.policy_dep_posteriors[0, 0, 4],
        [1 / 12, 1 / 12, 1 / 12, 3 / 4],
        rtol=1e-7,
        atol=1e-7,
    )
    assert np.argmax(agent.posterior_pi) == 20
    assert np.isclose(np.asarray(agent.posterior_pi, dtype=float).sum(), 1.0)
