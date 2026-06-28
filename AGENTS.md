# Repository Guidelines

## Project Structure & Module Organization

This repository reproduces HYDRA baselines for hierarchical text classification on WOS46985. Core Python modules live in `src/`: `src/data/` loads WOS datasets, `src/models/` defines HYDRA variants, and `src/utils/` contains metrics. Experiment entry points are in `scripts/`, including training, full experiment runs, setup, and result summarization. Raw datasets are expected under `data/wos_raw/`, local encoder files under `pretrained_models/scibert/`, generated model outputs under `results/`, logs under `logs/`, and reports under `docs/`.

## Build, Test, and Development Commands

- `uv sync`: install dependencies from `pyproject.toml` and `uv.lock`.
- `bash scripts/setup_server.sh`: prepare server-side data/model assets when missing.
- `bash scripts/run_hydra_wos.sh local 42`: run one HYDRA experiment; architecture may be `local`, `local_global`, or `local_nested`.
- `bash scripts/run_all_experiments.sh`: run all 3 architectures across the configured 5 seeds.
- `uv run python scripts/train_hydra.py --architecture local --seed 42`: run training directly with custom CLI flags.
- `uv run python scripts/summarize_results.py results`: regenerate `docs/experiment_results_summary.md`.

## Coding Style & Naming Conventions

Use Python 3.10+ and keep imports grouped as standard library, third-party, then local modules. Follow the existing style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, single-purpose modules, and explicit CLI argument names. Prefer `pathlib.Path` for filesystem paths and JSON files for run metadata. Keep generated artifacts out of source modules.

## Testing Guidelines

There is no dedicated test suite in the current tree. For code changes, run a focused smoke check such as:

```bash
uv run python scripts/train_hydra.py --num_epochs 1 --architecture local --seed 42
```

When changing metrics or dataset loading, validate against an existing run in `results/` and regenerate summaries. New tests, if added, should live under `tests/` and use `test_*.py` names.

## Commit & Pull Request Guidelines

Recent commits use Conventional Commit-style prefixes, for example `feat:`, `fix:`, and `rewrite:`. Keep commit messages imperative and scoped to one change. Pull requests should describe the experiment or code change, list commands run, note data/model prerequisites, and include key metric deltas or affected output paths when results change.

## Security & Configuration Tips

Do not commit large generated checkpoints, raw datasets, private logs, or local environment files. Treat `data/`, `pretrained_models/`, `results/`, and `logs/` as machine-local unless a specific artifact is intentionally documented.
