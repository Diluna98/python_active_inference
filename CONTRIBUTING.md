# Contributing

## Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Run the local checks before opening a pull request:

```bash
ruff check PyAIF tests
ruff format --check PyAIF tests
pytest
python -m build
python -m twine check dist/*
```

## Change policy

- Keep reusable inference and learning code under `PyAIF/`.
- Keep domain-specific generative models and likelihood construction under
  `examples/` or in an application repository.
- Add numerical regression tests for changes to inference, expected free
  energy, learning, or policy construction.
- Preserve structural zeros in categorical `A` and `B` updates.
- Do not add continuous-observation behavior to the `0.1.x` release line.

## Pull requests

Pull requests should describe the behavioral change, identify affected
examples, and include test evidence. CI tests Python 3.9 through 3.13 and builds
the wheel and source distribution.

## Releases

1. Update the version in `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Merge a pull request with all CI checks passing.
4. Create and publish a GitHub release for the matching version tag.
5. The protected `pypi` environment publishes with PyPI Trusted Publishing.

The repository owner must configure the PyPI trusted publisher and GitHub
`pypi` environment before the first release.
