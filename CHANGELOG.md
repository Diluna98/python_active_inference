# Changelog

## Unreleased

## 0.5.0 - 2026-11-11

### Added

- `GenerativeModel(B_dependencies=...)`: a factor's transition may depend on
  the previous states of other factors. `B[f]` then has shape
  `(states_dim[f], *[states_dim[d] for d in B_dependencies[f]], controls_dim[f])`.
  Supported by shallow, deep temporal, receding and filtered receding
  inference, Dirichlet `B` learning and transition novelty. Models without
  dependencies use the unchanged single-factor code path.
- `ActiveInfAgent(policy_cost_fn=...)`: an optional callback
  `policy_cost_fn(agent, policy_index, expected_states) -> float` whose value
  is added to each policy's expected free energy.
- Named expected-free-energy terms on every policy result:
  `expected_utility`, `state_information_gain`, `parameter_information_gain`
  and `policy_cost`, plus `PyAIF.expected_free_energy(...)`,
  `categorical_expected_utility` and `categorical_state_information_gain`.
- `PyAIF.transitions` with reusable prediction, message and evidence helpers.

### Fixed

- Shallow categorical inference negated the expected utility, so agents
  preferred the least preferred outcomes. Policy scores now use
  `expected_utility + state_information_gain + parameter_information_gain -
  policy_cost` in every mode.
- Parameter novelty (`pA`, `pB`, `pD`) entered the shallow and continuous deep
  scores with the wrong sign. All novelty methods now return non-negative
  information gain.
- Transition novelty paired the previous-state belief with the next-state axis
  of the Dirichlet weights. It now computes `sum q(s') W(s', s) q(s)`.
- Shallow and small-policy-set deep scoring ignored `modality_dependencies`
  when contracting likelihoods; modalities that depend on a subset of factors
  are now scored correctly in every mode.
- Novelty weights were computed from raw Dirichlet counts until
  `store_parameters()` was called manually. `reset()` now refreshes them, for
  continuous models as well.

### Changed (breaking)

- `ShallowPolicyInferenceResult.expected_free_energy` and
  `DeepPolicyInferenceResult.expected_free_energy` now hold the expected free
  energy `G` (lower is better). `agent.G_policy` and the value returned by
  `infer_policies()` remain the policy score `-G`.
- The result fields `risk`, `ambiguity` and `information_gain` are deprecated
  read-only aliases: `risk == -expected_utility`,
  `ambiguity == state_information_gain` and
  `information_gain == parameter_information_gain`.
- `deep_expected_free_energy`, `deep_categorical_policy_risk` and
  `deep_categorical_policy_ambiguity` are deprecated.

## 0.4.1 - 2026-09-18

### Performance

- Categorical deep-policy evaluation now lets NumPy optimize multi-factor
  contraction order, hoists likelihood entropy outside the future-time loop,
  and packs each policy-posterior batch once per timestep. Expected free
  energies and policy selection retain the existing semantics.
- Deterministic action selection treats posterior differences at numerical
  roundoff as ties and resolves them by stable policy order.

## 0.4.0 - 2026-09-16

### Added

- `FilteredRecedingHorizonInference`, which estimates the current hidden state
  once per observation and creates policy-conditioned future predictions only
  during policy evaluation.
- A runnable filter-then-plan example and regression coverage for categorical,
  continuous, multi-factor, serial, and parallel-policy configurations.

### Behavior

- The current posterior is independent of the available policy set. Future
  beliefs cannot send backward messages into that posterior.
- Policy evaluation retains the existing expected-free-energy terms and rolls
  the filtered belief forward through each policy's transition sequence.
- The filtered current posterior is carried directly. Policy-weighted future
  state averages are opt-in through `average_future_states=True`, avoiding
  unnecessary work in the default online-control path.
- Horizon and action-carry semantics match `RecedingHorizonInference`.
- Parameter learning remains unavailable for receding-horizon modes while
  overlapping-window evidence accounting is designed.

## 0.3.0 - 2026-09-16

### Added

- `RecedingHorizonInference`: reuse deep marginal message passing with a full
  relative planning window on every observation and select its first action.
- Action-conditioned belief carry-over and `observe(..., executed_action=...)`
  for controllers that override the selected action.
- Categorical and continuous observation support, absolute-time Monte Carlo
  seeds, parallel policy evaluation, lifecycle validation, and a runnable example.

### Compatibility

- `DeepTemporalInference` retains its existing fixed-window behavior.
- Horizon continues to count state time points: horizon 3 contains two actions.
- The new mode currently rejects parameter learning flags. Overlapping-window
  learning requires a separate evidence-accounting design.
- This is a planning lifecycle addition, not a performance optimization. Full
  lookahead is evaluated every cycle; latency can exceed fixed-window planning.

## 0.2.0 - 2026-07-24

### Added

- `ContinuousLikelihood` with scalar continuous modalities, per-modality state
  dependencies, observation grids, individual or joint log preferences, and
  optional domain learning and parameter-information-gain hooks.
- Continuous state and policy inference for both `ShallowInference` and
  `DeepTemporalInference`.
- Exact policy evaluation for small latent spaces and reproducible,
  vectorized Monte Carlo evaluation for larger spaces.
- `ContinuousLikelihood.from_model(...)` for adapting domain likelihood files
  without importing application code into PyAIF.
- A minimal Gaussian continuous-observation example and regression tests.

### Changed

- `ActiveInfAgent` now infers categorical versus continuous observation
  semantics from the likelihood component.
- Domain-specific continuous parameter updates are delegated through public
  callbacks instead of being hard-coded in the agent.

## 0.1.1 - 2026-07-24

### Added

- Configurable `policy_workers` for bounded parallel policy evaluation.
- Reproducible CPU benchmarks against classic NumPy pymdp and current JAX
  pymdp.

### Changed

- Deep temporal state inference, risk, and ambiguity evaluation now batch
  independent policies with NumPy.
- Shallow policy scoring no longer evaluates a disabled cost term, allowing
  policy sets larger than the number of hidden states.

## 0.1.0 - 2026-07-24

### Added

- Component-based `GenerativeModel`, `CategoricalLikelihood`,
  `ShallowInference`, and `DeepTemporalInference` APIs.
- Reusable shallow and deep categorical inference strategies.
- Reusable categorical parameter learning for `A`, `B`, `C`, `D`, and `E`.
- Numerical and end-to-end learning-example regression tests.
- Reproducible VFE-surprise animation for the uncertainty-learning example.
- Wheel validation, continuous integration, and guarded release automation.
- BSD-3-Clause licensing for the packaged software.

### Changed

- PyPI distribution name set to `pyaif-toolkit`; the import package remains
  `PyAIF`.
- Domain-specific likelihood construction moved out of the package.
- `ActiveInfAgent` now delegates reusable inference and learning behavior to
  focused modules.
- The retained learning-under-uncertainty example now uses the public component
  API and supports configurable trials, output directories, and random seeds.
- Deep expected-free-energy signs and cumulative initial-state learning retain
  the behavior of the validated research simulation.
- The obsolete handover and autonomous pick-and-place examples were removed.

### Compatibility

- The legacy agent constructor remains available while examples migrate.
- Continuous-observation support is planned for the `0.2.x` release line.
