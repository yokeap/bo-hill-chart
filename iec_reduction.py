"""Prototype: IEC unit-space / affinity-law reformulation.

Two questions, both answered against the real bf-gv data:

1. Does the affinity-law *similarity collapse* actually hold here? -> fig_iec_collapse
   (efficiency vs raw Q separates by head; vs unit discharge Q_ED it should merge).

2. Does unit space let us predict an *untested head*? -> leave-one-head-out CV
   comparing raw (Q,H,alpha) vs IEC unit features. This is the novelty claim:
   physically-grounded features turn head-extrapolation into interpolation.

    uv run python iec_reduction.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_all_data, METRIC
from src.gp_model import train_gp_model
from src.iec import unit_quantities, build_features, FEATURE_SETS
from src import viz  # sets IEC plot style on import


def _fit_predict(Xtr, ytr, Xte):
    """Standardise with TRAIN stats, fit GP, predict test."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    gp = train_gp_model((Xtr - mu) / sd, ytr)
    return gp.predict((Xte - mu) / sd)


def leave_one_head_out(df):
    """For each head: train on all OTHER heads, predict this head. Per-feature-set MAE."""
    y = df[METRIC].values
    heads = sorted(df["Head"].unique())
    rows = []
    for name, cols in FEATURE_SETS.items():
        X = build_features(df, cols)
        for h in heads:
            te = df["Head"].values == h
            tr = ~te
            pred = _fit_predict(X[tr], y[tr], X[te])
            rows.append({"features": name, "head": h,
                         "mae": float(np.mean(np.abs(y[te] - pred)))})
    return pd.DataFrame(rows)


def fig_collapse(df):
    """Efficiency vs raw Q (separates by head) vs unit Q_ED (should collapse)."""
    uq = unit_quantities(df)
    q, qed, eta = df["Dischargem"].values, uq["Q_ED"], df[METRIC].values
    heads = sorted(df["Head"].unique())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(heads)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(viz.COL2, viz.COL2 * 0.42))
    for h, c in zip(heads, cmap):
        m = df["Head"].values == h
        o1, o2 = np.argsort(q[m]), np.argsort(qed[m])
        a1.plot(q[m][o1], eta[m][o1], "-o", color=c, ms=3, label=f"{h:.0f} m")
        a2.plot(qed[m][o2], eta[m][o2], "-o", color=c, ms=3)
    a1.set(xlabel=r"Discharge $Q$ (m$^3$/s)", ylabel=r"Efficiency $\eta$",
           title="(a) Raw coordinates")
    a2.set(xlabel=r"Unit discharge $Q_{ED}\propto Q/\sqrt{gH}$", ylabel=r"Efficiency $\eta$",
           title="(b) IEC unit space (similarity)")
    a1.legend(title="Head", ncol=2, fontsize=6, title_fontsize=7, framealpha=0.9)
    for ax in (a1, a2):
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.4)
    fig.tight_layout()
    return fig


def main():
    from pathlib import Path
    out = Path("result"); out.mkdir(exist_ok=True)
    df = load_all_data("bf-gv")

    fig_collapse(df).savefig(out / "fig_iec_collapse.pdf")
    fig_collapse(df).savefig(out / "fig_iec_collapse.png")

    loho = leave_one_head_out(df)
    loho.to_csv(out / "iec_leave_one_head_out.csv", index=False)

    print(f"{'='*60}\nLeave-one-head-out MAE (predict an untested head)\n{'='*60}")
    piv = loho.pivot(index="head", columns="features", values="mae")
    piv = piv[list(FEATURE_SETS)]
    print(piv.round(4).to_string())
    print("\nmean:")
    print(piv.mean().round(4).to_string())

    best = piv.mean().idxmin()
    raw = piv["raw (Q,H,alpha)"].mean()
    imp = (raw - piv.mean().min()) / raw * 100
    print(f"\nBest feature set: {best}")
    print(f"Improvement over raw (mean MAE): {imp:.1f}%")
    print(f"\nSaved fig_iec_collapse + iec_leave_one_head_out.csv to {out}/")


if __name__ == "__main__":
    main()
