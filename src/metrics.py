"""
Metrics calculation utilities
"""

import numpy as np


def calculate_metrics(df, y_pred, metric='Overall Eff'):
    """Calculate prediction accuracy metrics"""
    y_true = df[metric].values
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2))
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}


def find_bep(df, sampled_indices, y_pred):
    """Find Best Efficiency Point from sampled data"""
    y_sampled = y_pred[sampled_indices]
    bep_idx = np.argmax(y_sampled)
    
    bep_data = df.iloc[sampled_indices[bep_idx]]
    return {
        'discharge': bep_data['Dischargem'],
        'head': bep_data['Head'],
        'efficiency': y_sampled[bep_idx],
        'gv': bep_data['G/V degree']
    }