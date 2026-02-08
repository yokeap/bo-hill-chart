"""
Random sampling strategy
"""

import numpy as np
from src.gp_model import train_gp_model


def simulate_random_sampling(df, n_samples=30, metric='Overall Eff', seed=42):
    """Random sampling baseline"""
    np.random.seed(seed)
    sampled_indices = np.random.choice(len(df), size=n_samples, replace=False).tolist()
    
    X = df[['Dischargem', 'Head', 'G/V degree']].values
    y = df[metric].values
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    
    X_train = X_normalized[sampled_indices]
    y_train = y[sampled_indices]
    gp = train_gp_model(X_train, y_train)
    y_pred, y_std = gp.predict(X_normalized, return_std=True)
    
    return sampled_indices, y_pred, y_std