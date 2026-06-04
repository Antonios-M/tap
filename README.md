# tap

[![CI](https://github.com/Antonios-M/tap/actions/workflows/ci.yml/badge.svg)](https://github.com/Antonios-M/tap/actions/workflows/ci.yml)
[![Release](https://github.com/Antonios-M/tap/actions/workflows/release.yml/badge.svg)](https://github.com/Antonios-M/tap/actions/workflows/release.yml)
[![Python](https://img.shields.io/badge/python->=3.13-blue.svg)](https://python.org)

> Traffic Simulator

## Development

```bash
# Clone and set up
git clone https://github.com/Antonios-M/tap
cd tap
uv sync

# Install pre-commit hooks (one-time)
uv run pre-commit install

# Run tests
uv run pytest

# Lint / format
uv run ruff check . --fix
uv run ruff format .

# Type check
uv run ty check

# Security scan
uv run bandit -c pyproject.toml -r tap/
```

## Releasing

Releases are fully automated. Simply merge PRs with
[Conventional Commits](https://www.conventionalcommits.org/) messages:

| Commit prefix | Version bump |
|---|---|
| `feat:` | minor (0.x.0) |
| `fix:`, `perf:` | patch (0.0.x) |
| `BREAKING CHANGE:` footer | major (x.0.0) |

On every merge to `main`, CI will open a release PR automatically.
Merge that PR to tag, publish, and update the changelog.
