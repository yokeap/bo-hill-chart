"""Bayesian optimization with UCB acquisition (mu + kappa*sigma).

Targets the *maximum* (BEP). Included to show it wins at BEP localization but
loses at global reconstruction -- the paper's central contrast.
"""
import numpy as np
from ..data_loader import feature_matrix
from ..gp_model import train_gp_model


def select_bo_ucb(df, n_samples, rng, kappa=2.0, n_initial=5):
    Xn, y = feature_matrix(df)
    chosen = [int(i) for i in rng.choice(len(df), size=n_initial, replace=False)]
    while len(chosen) < n_samples:
        gp = train_gp_model(Xn[chosen], y[chosen])
        mu, sigma = gp.predict(Xn, return_std=True)
        acq = mu + kappa * sigma
        acq[chosen] = -np.inf
        chosen.append(int(np.argmax(acq)))
    return chosen
