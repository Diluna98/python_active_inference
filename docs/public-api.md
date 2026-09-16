# Public API

PyAIF supports the component-based constructor:

```python
ActiveInfAgent(
    model=GenerativeModel(...),
    likelihood=CategoricalLikelihood(...),  # or ContinuousLikelihood(...)
    inference=ShallowInference(...),  # or a deep inference configuration
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

### Receding-horizon planning (0.3.0)

Use `RecedingHorizonInference(horizon=3, message_passing_iterations=10)`
for a full planning window at every observation. It accepts the same configuration
fields as `DeepTemporalInference` and supports categorical and continuous
observations. Horizon counts state time points, including the current state;
`horizon=3` therefore evaluates two-action policies.

```python
from PyAIF import ActiveInfAgent, RecedingHorizonInference

agent = ActiveInfAgent(
    model=model,
    likelihood=likelihood,
    inference=RecedingHorizonInference(horizon=3),
    action_selection="deterministic",
).reset()

for observation in observation_stream:
    agent.observe(observation)
    agent.infer_states()
    agent.infer_policies()
    action = agent.select_action()
    # Execute action and obtain the next observation.
```

Each decision reuses the existing policy-conditioned marginal message-passing
solver on relative indices `0..horizon-1`, scores the complete window using the
existing policy-value convention, and selects policy action zero. There is no
terminal-phase `None` action. The public clock advances once per selected action.
At the next observation, the prior is `B[action] @ q_current`, where `q_current`
is the policy-averaged belief at relative time zero, not a terminal prediction.
The generative model's initial-state prior is preserved for subsequent resets.

If the controller executes a different action, pass
`agent.observe(next_observation, executed_action=actual_action)`. Supply one
action index per hidden-state factor, including zero for uncontrolled factors.
If omitted, PyAIF assumes the returned action was executed once. Delayed,
partially executed, or continuous-duration controls need an application-specific
transition model; this API represents one discrete transition per decision.

Follow the lifecycle once per decision. Do not call `initialize_variables()`
between observations; window setup is automatic. `step_time()` is a compatibility
no-op in this mode. Optional explicit time indices must match the current
decision index. Adapters that constrain actions must examine policy row zero,
not `absolute_time % horizon`. Custom policies must have exactly `horizon - 1`
rows. Time-dependent categorical preferences use relative window indices.

This mode uses policy-conditioned temporal inference, not a new Bayesian filter
or a branching observation-contingent policy tree. Historical observations are
summarized in the carried prior, rather than retained for fixed-lag smoothing.
It is not guaranteed to be faster: all horizon points are evaluated every cycle
and exhaustive policy enumeration still grows exponentially with action depth.
Online parameter learning is currently rejected in this mode to avoid reusing
overlapping evidence. Existing shallow and fixed-window learning are unchanged.

Run `python examples/quickstart_receding.py` for a complete example.

### Common lifecycle

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
