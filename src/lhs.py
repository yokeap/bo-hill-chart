"""
Latin Hypercube Sampling strategy
"""

import numpy as np
from scipy.stats import qmc
from src.gp_model import train_gp_model


def simulate_lhs_sampling(df, n_samples=30, metric='Overall Eff', seed=42):
    """Latin Hypercube Sampling"""
    np.random.seed(seed)
    
    discharge_vals = df['Dischargem'].values
    head_vals = df['Head'].values
    
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    lhs_samples = sampler.random(n=n_samples)
    
    q_min, q_max = discharge_vals.min(), discharge_vals.max()
    h_min, h_max = head_vals.min(), head_vals.max()
    
    lhs_q = lhs_samples[:, 0] * (q_max - q_min) + q_min
    lhs_h = lhs_samples[:, 1] * (h_max - h_min) + h_min
    
    sampled_indices = []
    for q_target, h_target in zip(lhs_q, lhs_h):
        distances = np.sqrt((discharge_vals - q_target)**2 + (head_vals - h_target)**2)
        closest_idx = np.argmin(distances)
        if closest_idx not in sampled_indices:
            sampled_indices.append(closest_idx)
    
    while len(sampled_indices) < n_samples:
        available = [i for i in range(len(df)) if i not in sampled_indices]
        q_mid, h_mid = (q_min + q_max) / 2, (h_min + h_max) / 2
        center_distances = np.sqrt((discharge_vals[available] - q_mid)**2 + 
                                  (head_vals[available] - h_mid)**2)
        sampled_indices.append(available[np.argmin(center_distances)])
    
    X = df[['Dischargem', 'Head', 'G/V degree']].values
    y = df[metric].values
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    
    X_train = X_normalized[sampled_indices]
    y_train = y[sampled_indices]
    gp = train_gp_model(X_train, y_train)
    y_pred, y_std = gp.predict(X_normalized, return_std=True)
    
    return sampled_indices, y_pred, y_std