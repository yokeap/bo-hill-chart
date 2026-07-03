# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for a Q1-journal study on **experiment reduction for hydro-turbine hill charts**: given a real bulb/Kaplan turbine performance dataset, how few physical test points can reconstruct the full efficiency map (and locate the Best Efficiency Point, BEP) without losing accuracy. The comparison is between sampling strategies (Bayesian optimization vs. random/grid/LHS) fitted through a Gaussian Process surrogate.

Author is the domain owner (mechanical + data science). Treat contributions as PhD-level: methods must be defensible for peer review, not just runnable.

## Environment & commands

`uv`-managed (see `uv.lock`, `.python-version` = 3.9). Do not `pip install`; use `uv`.

```bash
uv sync                              # install deps into .venv
uv run python hill_chart.py          # traditional hill charts  -> general/*.png
uv run python bayesian_reduction.py  # sampling-method comparison -> result/fig*, summary.json
```

No test suite, linter, or CI configured. `main.py` is an empty stub — ignore it. There is no single-test command because there are no tests yet; if you add logic to `src/`, add a `test_*.py` beside it and run `uv run pytest <file>`.

## Data

- `bf-gv/` — **the dataset actually used.** One tidy CSV per head, named `<head>m.csv` (15m–22m). Columns: `G/V degree, Dischargem, Generator power, Mechanical power, Overall Eff, Mech Eff`. The loader parses head from the filename, adds a `Head` column, and drops rows where `Overall Eff >= 1` or `Dischargem` is NaN (placeholder rows). Yields ~72 clean experiments.
- `bf-bulb/` — older **wide/transposed** format (metrics as rows, GV angles as columns). Not consumed by the current loaders; different shape.
- `137-kw.xls` / `137-kw-bf-bulb.csv` — raw source export. Reference only.

Both `hill_chart.py` and `bayesian_reduction.py` carry their **own copy** of `load_all_data()` — keep them in sync, or (preferred) consolidate into `src/data_loader.py`.

## Architecture

**`hill_chart.py`** — standalone descriptive visualization (unchanged, not part of the study). Loads `bf-gv`, builds cubic-interpolated (`scipy.griddata`) 3D surface + 2D efficiency/power contours with constant-guide-vane curves, marks the BEP. Pure plotting. Carries its own `load_all_data`.

**The study lives in the `src/` package**; `bayesian_reduction.py` is a thin CLI over it. Data flow: load → z-score X=[Discharge, Head, GV] → each sampler picks row indices → fit one GP on those rows → predict all points → score.

- `src/data_loader.py` — `load_all_data`, `feature_matrix` (z-scored X, y), constants `FEATURES`, `METRIC`.
- `src/gp_model.py` — `train_gp_model` (ConstantKernel·Matern ν=2.5, `normalize_y`); `posterior_var` (analytic GP variance with a **fixed** kernel, used by ALM/ALC to score candidates cheaply).
- `src/samplers/` — each is `select(df, n_samples, rng) -> list[int]`. One-shot: `random`, `grid`, `lhs`. Sequential (refit GP per step): `bo_ucb` (UCB `μ+2σ`), `alm` (max σ), `alc` (min integrated variance). `_common.py` has nearest-row snapping + top-up. Registered in `SAMPLERS`.
- `src/metrics.py` — `calculate_metrics` (MAE/RMSE/MAPE/R² over all points), `evaluate` (fit+predict+score; **BEP read from the full-grid prediction**, not sampled rows), `ground_truth_bep`.
- `src/experiment.py` — `run_single`, `run_study` (sizes × seeds × methods → long DataFrame), `summarize` (mean±std). Per-(seed,method,n) RNG stream for reproducibility. Has a `__main__` self-check.
- `src/stats.py` — `pairwise_wilcoxon`: paired (over seeds) one-sided Wilcoxon signed-rank of a reference method vs. every other at a budget. Note the exact-test p-floor is 1/2^n_seeds.
- `src/viz.py` — IEEE style (Times ~8 pt, viridis, PDF). **`VIEW` = shared 3D view angle for all surfaces.** `surface_3d`, `contour_2d`, `sample_placement`, `learning_curves` (SEM bands).

CLI knobs at the top of `bayesian_reduction.py`: `SAMPLE_SIZES` (dense low-n sweep, default 6→30), `SEEDS` (default 10), `N_DEMO`/`DEMO_METHOD` (surface/placement figures, default ALC @ n=12).

## Results & interpretation

Two objectives need different samplers — this is the paper's thesis, confirmed on the real data:
- **Full-map reconstruction** (MAE/R² over all points): **ALM and ALC win at every budget**; BO-UCB is *worst* (clusters near the peak, starves the rest). See `fig_learning_curves` (a) and `fig_sample_placement`.
- **BEP localization**: **BO-UCB wins** (eff error ~1e-6 by n≈14); the others plateau ~1e-2. See `fig_learning_curves` (b).

Outputs in `result/`: `study_results.csv` (raw runs, for stats), `study_summary.csv` (mean±std), `table_n12.csv`, `wilcoxon_*.csv`, and figures `fig_ground_truth_{3d,2d}`, `fig_recon_{alc,boucb}_{3d,2d}`, `fig_goal_contrast` (the two-goals figure), `fig_error_2d`, `fig_sample_placement`, `fig_metrics_bars`, `fig_learning_curves`, plus `summary.json`.

**Companion docs for co-authors:** `PAPER_GUIDE.md` (plain-language guide + figure list + headline numbers) and `METHODS_ACTIVE_LEARNING.md` (GP/ALM/ALC/UCB math for the Methods section). Keep both in sync when results change.

## Conventions

- IEEE figures via `src/viz.py`; PNG (600 dpi) + PDF into `result/`. `hill_chart.py` writes its own style into `general/`.
- `result/` and `general/` outputs are committed and regenerate on every run — expect large binary diffs.
- ALC is O(candidates·steps) with a GP refit per step; a full `run_study` at the default sweep takes a couple of minutes.
