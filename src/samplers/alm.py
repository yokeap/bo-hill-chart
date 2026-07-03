"""Active Learning MacKay (ALM): sample where the GP is most uncertain.

Acquisition = posterior std. Pure exploration -> good global reconstruction.
"""
import numpy as np
from ..data_loader import feature_matrix
from ..gp_model import train_gp_model


def select_alm(df, n_samples, rng, n_initial=5):
    Xn, y = feature_matrix(df)
    chosen = [int(i) for i in rng.choice(len(df), size=n_initial, replace=False)]
    while len(chosen) < n_samples:
        gp = train_gp_model(Xn[chosen], y[chosen])
        _, sigma = gp.predict(Xn, return_std=True)
        sigma[chosen] = -np.inf
        chosen.append(int(np.argmax(sigma)))
    return chosen
