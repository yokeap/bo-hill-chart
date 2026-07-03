"""Active Learning Cohn (ALC): greedily add the point that most reduces the
GP's *integrated* predictive variance over the whole domain.

For each candidate we compute the mean posterior variance over all points if it
were added, with the fitted kernel held fixed, and pick the minimizer. This is
the theoretically-correct acquisition for full-surface reconstruction, and the
paper's recommended method.
"""
import numpy as np
from ..data_loader import feature_matrix
from ..gp_model import train_gp_model, posterior_var


def select_alc(df, n_samples, rng, n_initial=5):
    Xn, y = feature_matrix(df)
    n = len(df)
    chosen = [int(i) for i in rng.choice(n, size=n_initial, replace=False)]
    while len(chosen) < n_samples:
        gp = train_gp_model(Xn[chosen], y[chosen])
        kernel = gp.kernel_
        available = [i for i in range(n) if i not in chosen]
        best_i, best_score = None, np.inf
        for c in available:
            train_idx = chosen + [c]
            score = posterior_var(kernel, Xn[train_idx], Xn).mean()
            if score < best_score:
                best_score, best_i = score, c
        chosen.append(int(best_i))
    return chosen
