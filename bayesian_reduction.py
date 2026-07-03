"""Experiment-reduction study (thin CLI).

Runs every sampler over a range of sample sizes and seeds, then writes
learning curves, a summary table, and the raw results to result/.

    uv run python bayesian_reduction.py

Logic lives in the src/ package; see CLAUDE.md.
"""
import json
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # tiny-sample GP fits

import pandas as pd

from src.data_loader import load_all_data, METRIC
from src.metrics import ground_truth_bep
from src.experiment import run_study, run_single, summarize
from src.stats import pairwise_wilcoxon
from src import viz

SAMPLE_SIZES = [6, 8, 10, 12, 14, 16, 20, 25, 30]  # dense low-n regime
SEEDS = list(range(20))
N_DEMO = 12          # low budget used for the surface/placement figures
DEMO_METHOD = "ALC"  # reconstruction winner, used for the error map
RECON_DEMOS = ["ALC", "BO-UCB"]  # show both goals: map-builder vs optimum-seeker


def _save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")


def main():
    out = Path("result")
    out.mkdir(exist_ok=True)

    df = load_all_data("bf-gv")
    bep = ground_truth_bep(df)
    print(f"Loaded {len(df)} experiments from bf-gv")
    print(f"Ground-truth BEP: {bep}")

    print(f"\nRunning study: {len(SAMPLE_SIZES)} sizes x {len(SEEDS)} seeds x 6 methods...")
    results = run_study(df, SAMPLE_SIZES, SEEDS)
    summary = summarize(results)
    results.to_csv(out / "study_results.csv", index=False)
    summary.to_csv(out / "study_summary.csv", index=False)

    # ---- figures (IEEE style, shared 3D view angle) -----------------------
    print("Generating figures...")
    z_true = df[METRIC].values

    _save(viz.learning_curves(summary, n_seeds=len(SEEDS)), out, "fig_learning_curves")
    _save(viz.surface_3d(df, z_true, "Measured hill chart", bep=bep), out, "fig_ground_truth_3d")
    _save(viz.contour_2d(df, z_true, "Measured hill chart", bep=bep), out, "fig_ground_truth_2d")

    # reconstruction figures for each demo method (map-builder AND optimum-seeker)
    runs = {mth: run_single(df, mth, N_DEMO) for mth in RECON_DEMOS}
    for mth, (idx, m) in runs.items():
        tag = mth.lower().replace("-", "")
        _save(viz.surface_3d(df, m["y_pred"], f"{mth} reconstruction ($n={N_DEMO}$)",
                             sampled_idx=idx, bep=bep), out, f"fig_recon_{tag}_3d")
        _save(viz.contour_2d(df, m["y_pred"], f"{mth} reconstruction ($n={N_DEMO}$)",
                             sampled_idx=idx, bep=bep), out, f"fig_recon_{tag}_2d")

    # the money figure: same budget, two goals side by side
    panels = [(mth, runs[mth][1]["y_pred"], runs[mth][0], runs[mth][1]["mae"])
              for mth in RECON_DEMOS]
    _save(viz.goal_contrast(df, panels, bep), out, "fig_goal_contrast")

    idx, m = runs[DEMO_METHOD]
    z_err = abs(z_true - m["y_pred"])
    _save(viz.contour_2d(df, z_err, f"{DEMO_METHOD} reconstruction error ($n={N_DEMO}$)",
                         cmap="magma", cbar_label=r"$|\Delta\eta|$"), out, "fig_error_2d")
    _save(viz.metrics_bars(summary, N_DEMO, n_seeds=len(SEEDS)), out, "fig_metrics_bars")
    placements = {mth: run_single(df, mth, N_DEMO)[0] for mth in viz.COLORS}
    _save(viz.sample_placement(df, placements, z_true), out, "fig_sample_placement")

    # paper-ready comparison table at the demo budget
    tbl = summary[summary["n"] == N_DEMO].set_index("method")
    tbl = tbl.reindex(["ALM", "ALC", "Grid", "LHS", "Random", "BO-UCB"])
    tbl = tbl[["mae_mean", "rmse_mean", "r2_mean", "mape_mean",
               "bep_q_err_mean", "bep_h_err_mean", "bep_eff_err_mean"]].round(5)
    tbl.columns = ["MAE", "RMSE", "R2", "MAPE_%", "BEP_dQ", "BEP_dH", "BEP_deff"]
    tbl.insert(0, "reduction_%", round((1 - N_DEMO / len(df)) * 100, 1))
    tbl.to_csv(out / f"table_n{N_DEMO}.csv")

    # ---- Wilcoxon signed-rank significance (paired over seeds) -------------
    # reconstruction: are the active learners better? BEP: is BO-UCB better?
    headline = pd.concat([
        pairwise_wilcoxon(results, N_DEMO, "mae", "ALC"),
        pairwise_wilcoxon(results, N_DEMO, "mae", "ALM"),
        pairwise_wilcoxon(results, N_DEMO, "bep_eff_err", "BO-UCB"),
    ], ignore_index=True)
    headline.to_csv(out / f"wilcoxon_n{N_DEMO}.csv", index=False)

    # across-budget robustness: p-value of ALC vs each baseline (MAE) at every n
    sweep = pd.concat([pairwise_wilcoxon(results, nn, "mae", "ALC")
                       for nn in SAMPLE_SIZES], ignore_index=True)
    sweep.to_csv(out / "wilcoxon_mae_vs_n.csv", index=False)

    print(f"\n{'='*64}\nWilcoxon signed-rank at n={N_DEMO} (p<0.05 = significant)\n{'='*64}")
    for _, r in headline.iterrows():
        flag = "*" if r["significant"] else " "
        print(f"{flag} {r['reference']:>6} < {r['vs']:<7} [{r['metric']:>11}]  p={r['p_value']:.4f}")

    # headline table at the largest budget
    n = SAMPLE_SIZES[-1]
    print(f"\n{'='*64}\nResults at n={n} (mean over {len(SEEDS)} seeds)\n{'='*64}")
    print(f"{'Method':<10}{'MAE':>12}{'R2':>10}{'BEP eff err':>14}")
    top = summary[summary["n"] == n].sort_values("mae_mean")
    for _, r in top.iterrows():
        print(f"{r['method']:<10}{r['mae_mean']:>12.5f}{r['r2_mean']:>10.4f}{r['bep_eff_err_mean']:>14.6f}")

    json.dump({"n_experiments": len(df), "sample_sizes": SAMPLE_SIZES,
               "n_seeds": len(SEEDS), "bep_ground_truth": ground_truth_bep(df)},
              open(out / "summary.json", "w"), indent=2)
    print(f"\nSaved figures, study_results.csv, study_summary.csv to {out}/")


if __name__ == "__main__":
    main()
