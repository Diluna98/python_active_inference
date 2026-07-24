# PyAIF

PyAIF is a Python package for constructing discrete active-inference agents
with factorised hidden states. Version 0.1 provides categorical observations,
single-step inference, deep temporal inference, policy evaluation, action
selection, and categorical parameter learning.

Continuous-observation likelihoods are intentionally reserved for version 0.2.
Research applications and domain-specific likelihood construction belong in
the `examples/` directory or in separate repositories.

## Installation

PyAIF requires Python 3.9 or newer.

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Quick start

The public API separates the state-transition model, observation model, and
inference algorithm.

```python
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


# A[o, s]: categorical observation likelihood.
A = object_array(
    np.array([
        [0.95, 0.05],
        [0.05, 0.95],
    ])
)

# B[s_next, s_previous, action]: controlled transitions.
B_factor = np.zeros((2, 2, 2))
B_factor[:, :, 0] = np.eye(2)
B_factor[:, :, 1] = np.fliplr(np.eye(2))
B = object_array(B_factor)

# D[s]: initial-state prior. C[o]: outcome preferences.
D = object_array(np.array([0.5, 0.5]))
C = object_array(np.zeros(2))

model = GenerativeModel(
    B=B,
    D=D,
    controls_dim=[2],
    controllable_factors=[0],
)
likelihood = CategoricalLikelihood(
    A=A,
    preferences=C,
    _modality_dependencies=[[0]],
)
agent = ActiveInfAgent(
    model=model,
    likelihood=likelihood,
    inference=ShallowInference(),
    action_selection="deterministic",
)

agent.reset()
agent.observe([0])
agent.infer_states()
expected_free_energy, _ = agent.infer_policies()
action = agent.select_action()
```

Use `DeepTemporalInference(horizon=...)` for policy-dependent beliefs over
multiple time steps. Deep preferences have shape
`(number_of_outcomes, horizon)`.

## Agent lifecycle

The supported component-based lifecycle is:

```python
agent.reset(trial=trial)
agent.observe(observation, time_step=t)
agent.infer_states()
agent.infer_policies()
action = agent.select_action()
agent.learn()  # only when parameter learning is enabled
```

The older matrix-heavy constructor and methods such as `choose_action()` and
`perform_learning()` remain available while examples migrate, but new projects
should use the component constructor and lifecycle above.

## Model structure

- `GenerativeModel` owns state transitions (`B`), initial-state priors (`D`),
  control dimensions, controllable factors, and optional policies.
- `CategoricalLikelihood` owns observation likelihoods (`A`), preferences
  (`C`), and modality-to-factor dependencies.
- `ShallowInference` performs single-step factorised inference.
- `DeepTemporalInference` performs marginal message passing over a horizon.
- `PyAIF.learning` contains reusable categorical updates for `A`, `B`, `C`,
  `D`, and `E`.

See [Model shapes](docs/model-shapes.md) and
[Public API](docs/public-api.md) for details.

## Examples

- `examples/handover_task_single_agent/`: deep temporal handover agent.
- `examples/autonomus_picknplace/`: autonomous pick-and-place agent.
- `examples/learning_under_uncertinity/`: parameter-learning experiments.
- `examples/bounded_rationality/`: research code that will move to its own
  model-selection repository.

The automated regression suite covers the reusable component API and the
handover model/policy fixture.

Run the minimal component example with:

```bash
python examples/quickstart_discrete.py
```

## Version policy

- `0.1.x`: categorical observations and discrete parameter learning.
- `0.2.x`: planned continuous-observation likelihood components.

Behavioral changes are protected with numerical regression tests. Research
experiments may evolve independently from the packaged API.

## Development

```bash
ruff check PyAIF tests
ruff format --check PyAIF tests
pytest
python -m build
python -m twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the pull-request and release
process.
