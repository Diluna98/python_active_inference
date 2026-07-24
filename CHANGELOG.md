# Changelog

## 0.1.0 - Unreleased

### Added

- Component-based `GenerativeModel`, `CategoricalLikelihood`,
  `ShallowInference`, and `DeepTemporalInference` APIs.
- Reusable shallow and deep categorical inference strategies.
- Reusable categorical parameter learning for `A`, `B`, `C`, `D`, and `E`.
- Numerical and handover regression tests.
- Wheel validation, continuous integration, and guarded release automation.
- BSD-3-Clause licensing for the packaged software.

### Changed

- Domain-specific likelihood construction moved out of the package.
- `ActiveInfAgent` now delegates reusable inference and learning behavior to
  focused modules.

### Compatibility

- The legacy agent constructor remains available while examples migrate.
- Continuous-observation support is planned for the `0.2.x` release line.
