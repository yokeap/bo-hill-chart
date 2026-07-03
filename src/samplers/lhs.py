import numpy as np
from scipy.stats import qmc
from ._common import nearest_indices, fill_to


def select_lhs(df, n_samples, rng):
    """Latin Hypercube over (Q, H), snapped to nearest real rows."""
    q, h = df["Dischargem"].values, df["Head"].values
    seed = int(rng.integers(0, 2 ** 31 - 1))
    unit = qmc.LatinHypercube(d=2, seed=seed).random(n=n_samples)
    tq = unit[:, 0] * (q.max() - q.min()) + q.min()
    th = unit[:, 1] * (h.max() - h.min()) + h.min()
    chosen = nearest_indices(df, list(zip(tq, th)))
    return fill_to(df, chosen, n_samples, rng)
