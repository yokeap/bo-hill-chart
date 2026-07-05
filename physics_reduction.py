"""Prototype: does a physics-informed GP mean cut the experiments needed?

Ablation isolating the surrogate: same sampler (ALM) picks the points, then we
reconstruct with (a) a plain GP and (b) a GP on top of a hydraulic loss-model
mean. Lower MAE at low n = the physics prior buys experiment reduction.

    uv run python physics_reduction.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import load_all_data
from src.samplers import SAMPLERS
from src.metrics import evaluate
from src.physics_gp import evaluate_physics
from src.experiment import _rng
from src import viz

SIZES = [6, 8, 10, 12, 14, 16, 20, 25]
SEEDS = list(range(15))
SAMPLER = "ALM"


def run():
    df = load_all_data("bf-gv")
    select = SAMPLERS[SAMPLER]
    rows = []
    for n in SIZES:
        for s in SEEDS:
            idx = select(df, n, _rng(s, SAMPLER, n))
            rows.append({"n": n, "seed": s, "surrogate": "plain GP",
                         "mae": evaluate(df, idx)["mae"]})
            rows.append({"n": n, "seed": s, "surrogate": "physics-informed GP",
                         "mae": evaluate_physics(df, idx)["mae"]})
    return df, pd.DataFrame(rows)


def main():
    out = Path("result"); out.mkdir(exist_ok=True)
    df, res = run()
    res.to_csv(out / "physics_ablation.csv", index=False)

    g = res.groupby(["surrogate", "n"])["mae"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(viz.COL1, viz.COL1 * 0.8))
    for surr, color in [("plain GP", "#7f7f7f"), ("physics-informed GP", "#1b7837")]:
        sub = g[g["surrogate"] == surr].sort_values("n")
        sem = sub["std"] / np.sqrt(len(SEEDS))
        ax.plot(sub["n"], sub["mean"], "-o", color=color, ms=3, label=surr)
        ax.fill_between(sub["n"], np.maximum(sub["mean"] - sem, sub["mean"] * 0.3),
                        sub["mean"] + sem, color=color, alpha=0.18, linewidth=0)
    ax.set_yscale("log")
    ax.set(xlabel="Number of experiments $n$", ylabel="Reconstruction MAE",
           title=f"Physics-informed mean ({SAMPLER} sampling)")
    ax.grid(True, alpha=0.3, linestyle=":", which="both", linewidth=0.4)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "fig_physics_gp.pdf"); fig.savefig(out / "fig_physics_gp.png")

    piv = g.pivot(index="n", columns="surrogate", values="mean")
    piv["improvement_%"] = (piv["plain GP"] - piv["physics-informed GP"]) / piv["plain GP"] * 100
    print(f"{'='*60}\nReconstruction MAE: plain vs physics-informed GP\n{'='*60}")
    print(piv.round(5).to_string())
    print(f"\nMean improvement: {piv['improvement_%'].mean():.1f}%  "
          f"(low-n, n<=10: {piv.loc[piv.index <= 10, 'improvement_%'].mean():.1f}%)")


if __name__ == "__main__":
    main()
