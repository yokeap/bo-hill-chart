"""
Gaussian Process model utilities
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def train_gp_model(X_train, y_train):
    """
    Train Gaussian Process model
    
    Parameters:
    -----------
    X_train : array-like
        Training features (normalized)
    y_train : array-like
        Training targets
        
    Returns:
    --------
    GaussianProcessRegressor
        Trained GP model
    """
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=1e-6,
        normalize_y=True
    )
    gp.fit(X_train, y_train)
    
    return gp