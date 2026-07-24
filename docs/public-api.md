# Public API

PyAIF v0.1 supports the component-based constructor:

```python
ActiveInfAgent(
    model=GenerativeModel(...),
    likelihood=CategoricalLikelihood(...),
    inference=ShallowInference(...),  # or DeepTemporalInference(...)
)
```

All three components are required together.

## Stable components

### `GenerativeModel`

Contains the domain-independent hidden-state dynamics:

- `B`
- `D`
- `controls_dim`
- `controllable_factors`
- optional explicit `policies`

### `CategoricalLikelihood`

Contains:

- categorical likelihood tensors `A`
- outcome preferences
- modality-to-state-factor dependencies

Continuous likelihoods are not accepted in v0.1.

### `ShallowInference`

Configuration:

- `message_passing_iterations`
- `convergence_tolerance`
- `policy_workers` (default `1`)

State and policy calls expose diagnostics through
`agent.last_state_inference` and `agent.last_policy_inference`.

### `DeepTemporalInference`

Configuration:

- `horizon`
- `message_passing_iterations`
- `convergence_tolerance`
- `policy_workers` (default `1`)

Deep diagnostics include policy-dependent free energy, convergence, risk,
ambiguity, information gain, and the final policy posterior.

Deep inference automatically batches policies with NumPy. Set
`policy_workers` above one only after benchmarking a representative workload;
small models are normally faster with the default single batched worker.

## Lifecycle methods

- `reset(trial=0)`: normalize parameters and reset transient beliefs.
- `observe(observation, time_step=None)`: validate and store one multimodal
  observation.
- `infer_states()`: update hidden-state beliefs.
- `infer_policies()`: evaluate policies and update their posterior.
- `select_action()`: select an action and advance the component lifecycle time.
- `learn()`: apply enabled categorical parameter updates.

## Learning

Learning is opt-in through `learning_A`, `learning_B`, `learning_C`,
`learning_D`, and `learning_E`. The latest update summary is available as
`agent.last_learning`.

Pure learning helpers are exported from `PyAIF.learning` for applications that
manage their own trajectories.

## Compatibility layer

The legacy positional constructor remains available for current examples. It
is not the recommended interface for new projects and may be deprecated after
the examples complete their component-API migration.
