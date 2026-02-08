"""
Bayesian Optimization sampling strategy
"""

import numpy as np
from src.gp_model import train_gp_model


def simulate_bayesian_sampling(df, n_initial=5, n_iterations=25, metric='Overall Eff'):
    """
    Simulate Bayesian optimization with strategic initial sampling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    n_initial : int
        Number of strategic initial samples (corners + center)
    n_iterations : int
        Number of Bayesian optimization iterations
    metric : str
        Target metric to optimize
        
    Returns:
    --------
    tuple
        (sampled_indices, y_pred, y_std)
    """
    
    X = df[['Dischargem', 'Head', 'G/V degree']].values
    y = df[metric].values
    
    # Normalize inputs
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    
    print(f"{'='*70}")
    print(f"BAYESIAN OPTIMIZATION")
    print(f"{'='*70}")
    
    # Strategic initial sampling: corners + center
    discharge_vals = df['Dischargem'].values
    head_vals = df['Head'].values
    
    q_min, q_max = discharge_vals.min(), discharge_vals.max()
    h_min, h_max = head_vals.min(), head_vals.max()
    q_mid = (q_min + q_max) / 2
    h_mid = (h_min + h_max) / 2
    
    # Define corner and center points
    target_points = [
        (q_min, h_min),  # Bottom-left corner
        (q_max, h_min),  # Bottom-right corner
        (q_min, h_max),  # Top-left corner
        (q_max, h_max),  # Top-right corner
        (q_mid, h_mid),  # Center
    ]
    
    # Find closest actual data points to target points
    sampled_indices = []
    for target_q, target_h in target_points[:n_initial]:
        distances = np.sqrt((discharge_vals - target_q)**2 + (head_vals - target_h)**2)
        closest_idx = np.argmin(distances)
        if closest_idx not in sampled_indices:
            sampled_indices.append(closest_idx)
    
    # Fill if we need more initial points
    while len(sampled_indices) < n_initial:
        available = [i for i in range(len(df)) if i not in sampled_indices]
        center_distances = np.sqrt((discharge_vals[available] - q_mid)**2 + 
                                  (head_vals[available] - h_mid)**2)
        sampled_indices.append(available[np.argmin(center_distances)])
    
    print(f"Strategic initial: {n_initial} samples (corners + center)")
    
    # Bayesian optimization iterations
    for iteration in range(n_iterations):
        X_train = X_normalized[sampled_indices]
        y_train = y[sampled_indices]
        
        # Train GP model
        gp = train_gp_model(X_train, y_train)
        
        # Predict on all points
        mu, sigma = gp.predict(X_normalized, return_std=True)
        
        # UCB acquisition function
        kappa = 2.0
        acquisition = mu + kappa * sigma
        
        # Select next point (exclude already sampled)
        available = [i for i in range(len(df)) if i not in sampled_indices]
        next_idx = available[np.argmax(acquisition[available])]
        sampled_indices.append(next_idx)
    
    # Final GP model with all sampled points
    X_train_final = X_normalized[sampled_indices]
    y_train_final = y[sampled_indices]
    
    gp_final = train_gp_model(X_train_final, y_train_final)
    y_pred, y_std = gp_final.predict(X_normalized, return_std=True)
    
    return sampled_indices, y_pred, y_std