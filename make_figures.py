"""Generate the all-methods composite figures used in the manuscript.

Produces, at the headline budget n=12, one 2x3 panel each for:
  fig_all_methods_3d    reconstructed efficiency surfaces
  fig_all_methods_2d    reconstructed hill-chart contours
  fig_all_methods_error absolute reconstruction error |dEta|
plus fig_uncertainty (GP posterior sigma vs. actual error). The per-method and
learning-curve figures come from bayesian_reduction.py; this script only adds
the paper's six-up panels and the stopping-signal figure.

    uv run python make_figures.py
"""
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np
from scipy.stats import pearsonr

from src.data_loader import load_all_data, feature_matrix, METRIC
from src.gp_model import train_gp_model
from src.metrics import ground_truth_bep
from src.experiment import run_single
from src import viz

N_DEMO = 12


def main():
    out = Path("result"); out.mkdir(exist_ok=True)
    df = load_all_data("bf-gv")
    bep = ground_truth_bep(df)

    # {method: (y_pred, sampled_idx)} at the headline budget, seed 0
    preds = {}
    for mth in viz.METHOD_ORDER:
        idx, m = run_single(df, mth, N_DEMO)
        preds[mth] = (m["y_pred"], idx)

    figs = [
        ("fig_all_methods_3d", viz.all_methods_3d(df, preds)),
        ("fig_all_methods_2d", viz.all_methods_2d(df, preds, bep=bep)),
        ("fig_all_methods_error", viz.all_methods_error(df, preds)),
        ("fig_all_methods_error_3d", viz.all_methods_error_3d(df, preds)),
    ]

    # posterior uncertainty vs. actual error for ALC @ n=12 (stopping-signal fig)
    Xn, y = feature_matrix(df)
    idx = preds["ALC"][1]
    gp = train_gp_model(Xn[idx], y[idx])
    mu, sigma = gp.predict(Xn, return_std=True)
    err = np.abs(y - mu)
    r, p = pearsonr(sigma, err)
    print(f"sigma vs |dEta| at 72 points: Pearson r={r:.2f} (p={p:.1e})")
    figs.append(("fig_uncertainty", viz.uncertainty_error(df, sigma, err, idx)))

    for name, fig in figs:
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png")
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
