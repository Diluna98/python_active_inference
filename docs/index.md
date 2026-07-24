# PyAIF (`pyaif-toolkit`)

PyAIF is an **active-inference toolkit for Python**. It supports factorised
discrete hidden states with categorical or continuous observations, and
separates the generative model, likelihood, and temporal inference algorithm
into reusable components.

[Install PyAIF](#installation){ .md-button .md-button--primary }
[View source code](https://github.com/Diluna98/python_active_inference){ .md-button }

## Capabilities

- Categorical and continuous observation likelihoods
- Shallow single-step inference
- Deep temporal inference over policies
- Expected-free-energy policy evaluation and action selection
- Reusable categorical parameter-learning updates
- Optional domain-specific continuous-likelihood learning hooks
- Batched NumPy policy evaluation and bounded worker-thread support

## Installation

PyAIF requires Python 3.9 or newer. Install the published `pyaif-toolkit`
distribution from PyPI:

```bash
python -m pip install pyaif-toolkit
```

The distribution name is `pyaif-toolkit`; the Python import package is
`PyAIF`.

## Quick start

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


A = object_array(np.array([[0.95, 0.05], [0.05, 0.95]]))

B_factor = np.zeros((2, 2, 2))
B_factor[:, :, 0] = np.eye(2)
B_factor[:, :, 1] = np.fliplr(np.eye(2))
B = object_array(B_factor)

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
    modality_dependencies=[[0]],
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
expected_free_energy, policy_posterior = agent.infer_policies()
action = agent.select_action()
```

Use `DeepTemporalInference(horizon=...)` for policy-dependent beliefs across
multiple time steps. See the [public API](public-api.md) and
[model-shape reference](model-shapes.md) for component contracts and array
dimensions.

## Continuous observations

`ContinuousLikelihood` accepts a density function, observation grids, log
preferences, and the hidden-state factors on which each modality depends.
This lets applications supply domain-specific sensor models without changing
the reusable agent implementation.

```python
from PyAIF import ContinuousLikelihood

likelihood = ContinuousLikelihood(
    likelihood_fn=density,
    observation_grids=[observation_grid],
    log_preferences={0: log_preferences},
    modality_dependencies=[[0, 1]],
)
```

## Project links

- [PyPI package](https://pypi.org/project/pyaif-toolkit/)
- [GitHub repository](https://github.com/Diluna98/python_active_inference)
- [Issue tracker](https://github.com/Diluna98/python_active_inference/issues)
- [Citation information](citation.md)
- [BSD 3-Clause License](https://github.com/Diluna98/python_active_inference/blob/main/LICENSE)
