"""Gaussian-process surrogate and analytic posterior variance.

The variance helper lets active-learning samplers (ALM/ALC) score candidate
points with the *fitted* kernel held fixed -- the standard active-learning
setting, and far cheaper than re-optimizing hyperparameters per candidate.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

ALPHA = 1e-6  # observation noise / jitter for numerical stability


def train_gp_model(X_train, y_train, n_restarts=5):
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts,
        alpha=ALPHA, normalize_y=True,
    )
    gp.fit(X_train, y_train)
    return gp


def posterior_var(kernel, X_train, X_eval):
    """Predictive variance at X_eval given a training set, fixed kernel.

    var(x) = k(x,x) - k(x,S) [K(S,S)+aI]^-1 k(S,x),  evaluated per row of X_eval.
    """
    K = kernel(X_train) + ALPHA * np.eye(len(X_train))
    Ks = kernel(X_eval, X_train)              # (n_eval, n_train)
    v = np.linalg.solve(K, Ks.T)              # (n_train, n_eval)
    reduction = np.einsum("ij,ji->i", Ks, v)  # diag(Ks K^-1 Ks^T)
    return np.maximum(kernel.diag(X_eval) - reduction, 0.0)
