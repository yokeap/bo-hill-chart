"""
Grid sampling strategy
"""

import numpy as np
from src.gp_model import train_gp_model


def simulate_grid_sampling(df, n_samples=30, metric='Overall Eff'):
    """Grid-based sampling"""
    discharge_vals = df['Dischargem'].values
    head_vals = df['Head'].values
    
    n_per_axis = int(np.sqrt(n_samples))
    q_grid = np.linspace(discharge_vals.min(), discharge_vals.max(), n_per_axis)
    h_grid = np.linspace(head_vals.min(), head_vals.max(), n_per_axis)
    
    sampled_indices = []
    for q_target in q_grid:
        for h_target in h_grid:
            if len(sampled_indices) >= n_samples:
                break
            distances = np.sqrt((discharge_vals - q_target)**2 + (head_vals - h_target)**2)
            closest_idx = np.argmin(distances)
            if closest_idx not in sampled_indices:
                sampled_indices.append(closest_idx)
        if len(sampled_indices) >= n_samples:
            break
    
    X = df[['Dischargem', 'Head', 'G/V degree']].values
    y = df[metric].values
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    
    X_train = X_normalized[sampled_indices]
    y_train = y[sampled_indices]
    gp = train_gp_model(X_train, y_train)
    y_pred, y_std = gp.predict(X_normalized, return_std=True)
    
    return sampled_indices, y_pred, y_std