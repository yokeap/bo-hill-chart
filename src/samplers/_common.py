"""Shared helpers for samplers."""
import numpy as np


def nearest_indices(df, targets, k=None):
    """Greedy: for each (q, h) target pick the closest not-yet-used real row."""
    q = df["Dischargem"].values
    h = df["Head"].values
    chosen = []
    for tq, th in targets:
        d = np.sqrt((q - tq) ** 2 + (h - th) ** 2)
        order = np.argsort(d)
        for idx in order:
            if idx not in chosen:
                chosen.append(int(idx))
                break
        if k is not None and len(chosen) >= k:
            break
    return chosen


def fill_to(df, chosen, n, rng):
    """Top up a selection to n with random unused rows (keeps counts equal)."""
    if len(chosen) >= n:
        return chosen[:n]
    pool = [i for i in range(len(df)) if i not in chosen]
    extra = rng.choice(pool, size=n - len(chosen), replace=False)
    return chosen + [int(i) for i in extra]
