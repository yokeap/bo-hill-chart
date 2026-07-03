import numpy as np
from ._common import nearest_indices, fill_to


def select_grid(df, n_samples, rng):
    """Deterministic space-filling: snap a regular Q x H grid to nearest rows."""
    q, h = df["Dischargem"].values, df["Head"].values
    n_axis = max(2, int(round(np.sqrt(n_samples))))
    qs = np.linspace(q.min(), q.max(), n_axis)
    hs = np.linspace(h.min(), h.max(), n_axis)
    targets = [(tq, th) for tq in qs for th in hs]
    chosen = nearest_indices(df, targets)
    return fill_to(df, chosen, n_samples, rng)
