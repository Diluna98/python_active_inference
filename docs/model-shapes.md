# Model shapes

PyAIF represents each collection of factor- or modality-specific arrays
as a one-dimensional NumPy object array.

## Hidden-state factors

For hidden-state factor `f`:

- `D[f]` has shape `(number_of_states_f,)`.
- `B[f]` has shape
  `(number_of_next_states_f, number_of_previous_states_f, number_of_actions_f)`.
- `controls_dim[f]` equals `number_of_actions_f`.

The first two axes of `B[f]` therefore use the convention
`B[next_state, previous_state, action]`.

### Transitions that depend on other factors

`GenerativeModel(B_dependencies=...)` lets the next state of factor `f`
depend on the previous states of several factors. `B_dependencies[f]` lists
those factors, normally including `f` itself:

```text
B_dependencies[f] == [f, g]
B[f].shape == (states_dim[f], states_dim[f], states_dim[g], controls_dim[f])
P(s_f[t+1] | s_f[t], s_g[t], u_f[t]) == B[f][:, s_f[t], s_g[t], u_f[t]]
```

Each `B[f][:, ..., action]` is normalised over its first (next-state) axis.
Beliefs stay factorised: predictions contract `B[f]` with the belief of
every listed factor, and a factor receives smoothing messages from every
transition that depends on it. Omitting `B_dependencies` is equivalent to
`[[0], [1], ...]`.

## Observation modalities

For observation modality `m`:

- `A[m]` has one observation axis followed by the hidden-state axes on which
  that modality depends.
- `modality_dependencies[m]` lists those factors in the same order.

For example, a modality with three outcomes depending on factors 0 and 2 has:

```text
A[m].shape == (3, states_dim[0], states_dim[2])
modality_dependencies[m] == [0, 2]
```

Each conditional distribution is normalized over the first, observation axis.

For continuous modality `m`:

- `likelihood_fn(value, m)` returns a nonnegative density tensor whose axes
  follow `modality_dependencies[m]`.
- `observation_grids[m]` is a strictly increasing one-dimensional array.
- `log_preferences[m]` has the same length as that grid.
- a joint preference keyed by `(m, n)` has shape
  `(len(observation_grids[m]), len(observation_grids[n]))`.

Continuous observations are scalar values, while hidden states remain
categorical and factorised.

## Preferences

- Shallow inference: `C[m].shape == (number_of_outcomes_m,)`.
- Deep temporal inference:
  `C[m].shape == (number_of_outcomes_m, temporal_horizon)`.

PyAIF converts preference parameters to log preferences during agent reset.

## Policies

A policy is an integer array with shape:

```text
(number_of_action_steps, number_of_state_factors)
```

For deep inference, `number_of_action_steps` is normally
`temporal_horizon - 1`. Uncontrollable factors use action index zero.

## Structural zeros and learning

Zero-valued entries in `A` and `B` are treated as structural zeros by the
categorical learner. Learning updates supported entries but does not introduce
probability mass into impossible mappings or transitions.
