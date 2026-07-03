"""Wilcoxon signed-rank significance tests over seeds.

Paired by seed (each seed is a matched replication): at a fixed budget n, test
whether a reference method's metric is significantly *lower* (better, for our
error metrics) than each competitor's. One-sided, exact p-values.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def pairwise_wilcoxon(results, n, metric, reference, alternative="less"):
    """reference vs every other method at budget n, on `metric` (lower=better).

    alternative='less' tests median(reference - other) < 0, i.e. reference wins.
    Returns a tidy DataFrame with medians, p-value, and a significance flag.
    """
    pivot = results[results["n"] == n].pivot(index="seed", columns="method", values=metric)
    ref = pivot[reference]
    rows = []
    for m in pivot.columns:
        if m == reference:
            continue
        diff = (ref - pivot[m]).values
        if np.allclose(diff, 0):
            stat, p = np.nan, 1.0          # identical -> no evidence of difference
        else:
            stat, p = wilcoxon(ref, pivot[m], alternative=alternative)
        rows.append({
            "metric": metric, "n": n, "reference": reference, "vs": m,
            "median_ref": float(ref.median()), "median_vs": float(pivot[m].median()),
            "p_value": float(p), "significant": bool(p < 0.05),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # ponytail: runnable self-check against a saved study
    res = pd.read_csv("result/study_results.csv")
    out = pairwise_wilcoxon(res, 12, "mae", "ALC")
    assert set(out["vs"]) == {"ALM", "Grid", "LHS", "Random", "BO-UCB"}
    assert (out["p_value"] >= 1 / 2 ** res["seed"].nunique() - 1e-9).all()
    print(out.to_string(index=False))
    print("\nOK: Wilcoxon runs; p-values respect the exact-test floor.")
