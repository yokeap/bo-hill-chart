"""Run the full study: every sampler x every sample size x every seed.

Produces a long-form DataFrame (one row per run) for learning curves, plus a
mean+/-std summary. Replication over seeds is what makes the comparison
publishable rather than anecdotal.
"""
import numpy as np
import pandas as pd
from .data_loader import load_all_data
from .metrics import evaluate
from .samplers import SAMPLERS


def _rng(seed, method, n):
    """Independent, reproducible stream per (seed, method, n)."""
    tag = sum(ord(c) for c in method)
    return np.random.default_rng([int(seed), tag, int(n)])


def run_single(df, method, n, seed=0):
    """One run: return (sampled indices, evaluate() dict) for figure generation."""
    idx = SAMPLERS[method](df, n, _rng(seed, method, n))
    return idx, evaluate(df, idx)


def run_study(df, sample_sizes, seeds, samplers=SAMPLERS):
    rows = []
    for method, select in samplers.items():
        for n in sample_sizes:
            for seed in seeds:
                idx = select(df, n, _rng(seed, method, n))
                assert len(set(idx)) == n, f"{method}: expected {n} unique, got {len(set(idx))}"
                m = evaluate(df, idx)
                rows.append({
                    "method": method, "n": n, "seed": seed,
                    "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"], "mape": m["mape"],
                    "bep_q_err": m["bep_error"]["discharge"],
                    "bep_h_err": m["bep_error"]["head"],
                    "bep_eff_err": m["bep_error"]["efficiency"],
                })
    return pd.DataFrame(rows)


def summarize(results):
    """Mean +/- std over seeds, per (method, n)."""
    num = results.drop(columns=["seed"])
    agg = num.groupby(["method", "n"]).agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


if __name__ == "__main__":
    # ponytail: runnable self-check on a tiny study, not a full run
    df = load_all_data("bf-gv")
    res = run_study(df, sample_sizes=[8, 16], seeds=[0, 1])
    assert set(res["method"]) == set(SAMPLERS)
    assert res["r2"].max() <= 1.0 + 1e-9
    assert res[["mae", "rmse", "mape"]].notna().all().all()
    print(summarize(res).to_string(index=False))
    print("\nOK: study runs, metrics finite, r2<=1.")
