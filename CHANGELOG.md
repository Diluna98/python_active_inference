# Changelog

## Unreleased

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
