# Public API

PyAIF supports the component-based constructor:

```python
ActiveInfAgent(
    model=GenerativeModel(...),
    likelihood=CategoricalLikelihood(...),  # or ContinuousLikelihood(...)
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

### `ContinuousLikelihood`

Contains:

- a scalar density callback for each modality
- one numerical integration grid per modality
- individual or joint log preferences on those grids
- modality-to-state-factor dependencies
- an optional vectorized grid-density callback
- optional likelihood-learning, preference-learning, and
  parameter-information-gain callbacks

`ContinuousLikelihood.from_model(...)` adapts a domain object that exposes
`likelihoods`, `get_o_grid`, and `log_preferences`.

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
- `learn()`: apply enabled structural or likelihood parameter updates.

## Learning

Learning is opt-in through `learning_A`, `learning_B`, `learning_C`,
`learning_D`, and `learning_E`. The latest update summary is available as
`agent.last_learning`.

Categorical likelihood parameters use the built-in Dirichlet learner.
Continuous likelihood parameters are domain-dependent and use the
`ContinuousLikelihood.learning_fn` callback. Structural `B`, `D`, and `E`
learning remains available with continuous observations.

## Compatibility layer

The legacy positional constructor remains available for current examples. It
is not the recommended interface for new projects and may be deprecated after
the examples complete their component-API migration.
