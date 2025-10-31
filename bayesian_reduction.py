import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

def load_all_data(folder_path='bf-gv'):
    """Load all CSV files and combine them with head information"""
    folder = Path(folder_path)
    all_data = []
    
    csv_files = sorted(folder.glob('*.csv'))
    
    for file in csv_files:
        head = float(file.stem.replace('m', ''))
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df['Head'] = head
        df = df[(df['Overall Eff'] < 1) & (df['Dischargem'].notna())]
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def train_gp_model(X_train, y_train):
    """Train Gaussian Process model"""
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                  alpha=1e-6, normalize_y=True)
    gp.fit(X_train, y_train)
    return gp

def simulate_bayesian_sampling(df, n_initial=5, n_iterations=25, metric='Overall Eff'):
    """Simulate Bayesian optimization with strategic initial sampling"""
    
    X = df[['Dischargem', 'Head', 'G/V degree']].values
    y = df[metric].values
    
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    
    print(f"\n{'='*70}")
    print(f"BAYESIAN OPTIMIZATION")
    print(f"{'='*70}")
    
    # Strategic initial sampling: corners + center
    discharge_vals = df['Dischargem'].values
    head_vals = df['Head'].values
    
    q_min, q_max = discharge_vals.min(), discharge_vals.max()
    h_min, h_max = head_vals.min(), head_vals.max()
    q_mid = (q_min + q_max) / 2
    h_mid = (h_min + h_max) / 2
    
    target_points = [
        (q_min, h_min), (q_max, h_min),
        (q_min, h_max), (q_max, h_max),
        (q_mid, h_mid)
    ]
    
    sampled_indices = []
    for target_q, target_h in target_points[:n_initial]:
        distances = np.sqrt((discharge_vals - target_q)**2 + (head_vals - target_h)**2)
        closest_idx = np.argmin(distances)
        if closest_idx not in sampled_indices:
            sampled_indices.append(closest_idx)
    
    while len(sampled_indices) < n_initial:
        available = [i for i in range(len(df)) if i not in sampled_indices]
        center_distances = np.sqrt((discharge_vals[available] - q_mid)**2 + 
                                  (head_vals[available] - h_mid)**2)
        sampled_indices.append(available[np.argmin(center_distances)])
    
    print(f"Strategic initial: {n_initial} samples (corners + center)")
    
    # Bayesian iterations
    for iteration in range(n_iterations):
        X_train = X_normalized[sampled_indices]
        y_train = y[sampled_indices]
        
        gp = train_gp_model(X_train, y_train)
        mu, sigma = gp.predict(X_normalized, return_std=True)
        
        kappa = 2.0
        acquisition = mu + kappa * sigma
        
        available = [i for i in range(len(df)) if i not in sampled_indices]
        next_idx = available[np.argmax(acquisition[available])]
        sampled_indices.append(next_idx)
    
    # Final model
    X_train_final = X_normalized[sampled_indices]
    y_train_final = y[sampled_indices]
    gp_final = train_gp_model(X_train_final, y_train_final)
    y_pred, y_std = gp_final.predict(X_normalized, return_std=True)
    
    return sampled_indices, y_pred, y_std

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

def create_ground_truth_3d(df, metric='Overall Eff', figsize=(8, 6)):
    """Ground truth 3D surface"""
    discharge = df['Dischargem'].values
    head = df['Head'].values
    efficiency = df[metric].values
    
    bep_idx = efficiency.argmax()
    bep_info = (discharge[bep_idx], head[bep_idx], efficiency[bep_idx], df.iloc[bep_idx]['G/V degree'])
    
    discharge_grid = np.linspace(discharge.min(), discharge.max(), 100)
    head_grid = np.linspace(head.min(), head.max(), 100)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    E_grid = griddata((discharge, head), efficiency, (Q_grid, H_grid), method='cubic')
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(Q_grid, H_grid, E_grid, cmap='RdYlGn', alpha=0.85,
                          edgecolor='none', antialiased=True, shade=True)
    ax.scatter(discharge, head, efficiency, c='blue', marker='o', s=25,
              edgecolors='darkblue', linewidths=0.5, alpha=0.7, label='Experimental data')
    ax.scatter([bep_info[0]], [bep_info[1]], [bep_info[2]], c='red', marker='*',
              s=400, edgecolors='darkred', linewidths=2, label='BEP', zorder=10)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.08)
    cbar.set_label('Efficiency', fontsize=11)
    
    ax.set_xlabel('Discharge, Q (m^3/s)', fontsize=11, labelpad=8)
    ax.set_ylabel('Head, H (m)', fontsize=11, labelpad=8)
    ax.set_zlabel('Efficiency', fontsize=11, labelpad=8)
    ax.set_title(f'Ground Truth Surface\n({len(df)} experiments)', fontsize=12, pad=15)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.view_init(elev=18, azim=-50, roll=0)
    ax.dist = 11
    
    plt.tight_layout()
    return fig, bep_info

def create_method_comparison_surfaces(df, method_results, metric='Overall Eff', figsize=(12, 8)):
    """Compare all sampling methods"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    discharge = df['Dischargem'].values
    head = df['Head'].values
    
    methods = ['Bayesian', 'Random', 'Grid', 'LHS']
    colors = ['lime', 'red', 'blue', 'orange']
    
    for idx, (method, color) in enumerate(zip(methods, colors)):
        ax = axes[idx // 2, idx % 2]
        
        sampled_indices = method_results[method]['indices']
        y_pred = method_results[method]['predictions']
        
        discharge_grid = np.linspace(discharge.min(), discharge.max(), 150)
        head_grid = np.linspace(head.min(), head.max(), 150)
        Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
        
        discharge_sampled = discharge[sampled_indices]
        head_sampled = head[sampled_indices]
        y_sampled = y_pred[sampled_indices]
        E_pred = griddata((discharge_sampled, head_sampled), y_sampled,
                         (Q_grid, H_grid), method='cubic')
        
        levels = np.linspace(np.nanmin(E_pred), np.nanmax(E_pred), 20)
        cf = ax.contourf(Q_grid, H_grid, E_pred, levels=levels, cmap='RdYlGn', alpha=0.9)
        
        ax.scatter(discharge_sampled, head_sampled, c=color, marker='o', s=60,
                  edgecolors='black', linewidths=1.5, alpha=0.9, zorder=5)
        
        mae = method_results[method]['mae']
        r2 = method_results[method]['r2']
        
        ax.set_xlabel('Discharge, Q (m^3/s)', fontsize=10)
        ax.set_ylabel('Head, H (m)', fontsize=10)
        ax.set_title(f'{method} Sampling\nMAE: {mae:.4f}, R²: {r2:.4f}',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
        plt.colorbar(cf, ax=ax, pad=0.02)
    
    plt.tight_layout()
    return fig

def create_method_comparison_metrics(method_results, figsize=(10, 6)):
    """Bar chart comparing method performance"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    methods = ['Bayesian', 'Random', 'Grid', 'LHS']
    colors = ['lime', 'red', 'blue', 'orange']
    
    # MAE comparison
    ax1 = axes[0]
    maes = [method_results[m]['mae'] for m in methods]
    bars1 = ax1.bar(methods, maes, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    ax1.set_title('Prediction Error Comparison\n(Lower is Better)',
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # R² comparison
    ax2 = axes[1]
    r2s = [method_results[m]['r2'] for m in methods]
    bars2 = ax2.bar(methods, r2s, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax2.set_title('Prediction Accuracy Comparison\n(Higher is Better)',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig

def create_bep_comparison_all_methods(bep_ground_truth, method_results, figsize=(12, 5)):
    """BEP comparison for all methods"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    methods = ['Bayesian', 'Random', 'Grid', 'LHS']
    colors = ['lime', 'red', 'blue', 'orange']
    
    # Left: BEP locations
    ax1 = axes[0]
    ax1.scatter(bep_ground_truth[0], bep_ground_truth[1], c='gold', marker='*', s=800,
               edgecolors='black', linewidths=3, label='Ground Truth', zorder=10)
    
    for method, color in zip(methods, colors):
        bep = method_results[method]['bep']
        ax1.scatter(bep['discharge'], bep['head'], c=color, marker='o', s=300,
                   edgecolors='black', linewidths=2, label=method, zorder=5, alpha=0.8)
    
    ax1.set_xlabel('Discharge, Q (m^3/s)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Head, H (m)', fontsize=11, fontweight='bold')
    ax1.set_title('BEP Location Comparison', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(fontsize=9, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right: BEP errors
    ax2 = axes[1]
    discharge_errors = [method_results[m]['bep_error']['discharge'] for m in methods]
    head_errors = [method_results[m]['bep_error']['head'] for m in methods]
    eff_errors = [method_results[m]['bep_error']['efficiency'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax2.bar(x - width, discharge_errors, width, label='Discharge Error',
           color='steelblue', edgecolor='black', linewidth=1)
    ax2.bar(x, head_errors, width, label='Head Error',
           color='coral', edgecolor='black', linewidth=1)
    ax2.bar(x + width, eff_errors, width, label='Efficiency Error',
           color='lightgreen', edgecolor='black', linewidth=1)
    
    ax2.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
    ax2.set_title('BEP Prediction Error by Method', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.legend(fontsize=9, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    return fig

def create_bep_comparison_table_all_methods(bep_ground_truth, method_results, figsize=(12, 4)):
    """Table showing BEP comparison for all methods"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    methods = ['Ground Truth', 'Bayesian', 'Random', 'Grid', 'LHS']
    
    table_data = [['Method', 'Discharge (m^3/s)', 'Head (m)', 'Efficiency', 'Q Error', 'H Error', 'Eff Error']]
    
    # Ground truth row
    table_data.append([
        'Ground Truth',
        f"{bep_ground_truth[0]:.4f}",
        f"{bep_ground_truth[1]:.2f}",
        f"{bep_ground_truth[2]:.4f}",
        '-',
        '-',
        '-'
    ])
    
    # Each method row
    for method in ['Bayesian', 'Random', 'Grid', 'LHS']:
        bep = method_results[method]['bep']
        error = method_results[method]['bep_error']
        
        table_data.append([
            method,
            f"{bep['discharge']:.4f}",
            f"{bep['head']:.2f}",
            f"{bep['efficiency']:.4f}",
            f"{error['discharge']:.4f}",
            f"{error['head']:.2f}",
            f"{error['efficiency']:.4f}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.12, 0.14, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(7):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style ground truth row
    for i in range(7):
        table[(1, i)].set_facecolor('#FFD700')
        table[(1, i)].set_text_props(weight='bold')
    
    # Find best method (lowest total error)
    total_errors = []
    for method in ['Bayesian', 'Random', 'Grid', 'LHS']:
        error = method_results[method]['bep_error']
        total = error['discharge'] + error['head'] + error['efficiency']
        total_errors.append(total)
    
    best_method_idx = total_errors.index(min(total_errors)) + 2
    
    # Highlight best method
    for i in range(7):
        table[(best_method_idx, i)].set_facecolor('#90EE90')
        table[(best_method_idx, i)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(2, 6):
        if i != best_method_idx:
            for j in range(7):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('BEP Comparison: All Methods\n(Best method highlighted in green)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def create_bayesian_3d(df, sampled_indices, y_pred, bep_ground_truth, metric='Overall Eff', figsize=(8, 6)):
    """Bayesian prediction 3D surface"""
    discharge = df['Dischargem'].values
    head = df['Head'].values
    
    discharge_sampled = discharge[sampled_indices]
    head_sampled = head[sampled_indices]
    y_sampled = y_pred[sampled_indices]
    
    # Find Bayesian BEP
    bep_idx = np.argmax(y_sampled)
    bayesian_bep = {
        'discharge': discharge_sampled[bep_idx],
        'head': head_sampled[bep_idx],
        'efficiency': y_sampled[bep_idx],
        'gv': df.iloc[sampled_indices[bep_idx]]['G/V degree']
    }
    
    # Create grid
    discharge_grid = np.linspace(discharge.min(), discharge.max(), 100)
    head_grid = np.linspace(head.min(), head.max(), 100)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    E_grid = griddata((discharge_sampled, head_sampled), y_sampled, (Q_grid, H_grid), method='cubic')
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(Q_grid, H_grid, E_grid, cmap='RdYlGn', alpha=0.85,
                          edgecolor='none', antialiased=True, shade=True)
    
    ax.scatter(discharge_sampled, head_sampled, y_sampled, c='red', marker='o', s=60,
              edgecolors='darkred', linewidths=1.5, alpha=0.9, label='Sampled experiments')
    
    ax.scatter([bep_ground_truth[0]], [bep_ground_truth[1]], [bep_ground_truth[2]],
              c='gold', marker='*', s=400, edgecolors='black', linewidths=2,
              label='BEP (ground truth)', zorder=10)
    
    ax.scatter([bayesian_bep['discharge']], [bayesian_bep['head']], [bayesian_bep['efficiency']],
              c='lime', marker='*', s=400, edgecolors='black', linewidths=2,
              label='BEP (Bayesian)', zorder=10)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.08)
    cbar.set_label('Efficiency', fontsize=11)
    
    ax.set_xlabel('Discharge, Q (m^3/s)', fontsize=11, labelpad=8)
    ax.set_ylabel('Head, H (m)', fontsize=11, labelpad=8)
    ax.set_zlabel('Efficiency', fontsize=11, labelpad=8)
    ax.set_title(f'Bayesian Prediction\n({len(sampled_indices)} experiments)', fontsize=12, pad=15)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.view_init(elev=18, azim=-50, roll=0)
    ax.dist = 11
    
    plt.tight_layout()
    return fig

def create_error_3d(df, sampled_indices, y_pred, metric='Overall Eff', figsize=(8, 6)):
    """Error surface 3D"""
    discharge = df['Dischargem'].values
    head = df['Head'].values
    y_true = df[metric].values
    
    discharge_grid = np.linspace(discharge.min(), discharge.max(), 100)
    head_grid = np.linspace(head.min(), head.max(), 100)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    
    E_true = griddata((discharge, head), y_true, (Q_grid, H_grid), method='cubic')
    
    discharge_sampled = discharge[sampled_indices]
    head_sampled = head[sampled_indices]
    y_sampled = y_pred[sampled_indices]
    E_pred = griddata((discharge_sampled, head_sampled), y_sampled, (Q_grid, H_grid), method='cubic')
    
    E_error = np.abs(E_true - E_pred)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(Q_grid, H_grid, E_error, cmap='Reds', alpha=0.85,
                          edgecolor='none', antialiased=True, shade=True)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.08)
    cbar.set_label('Absolute Error', fontsize=11)
    
    ax.set_xlabel('Discharge, Q (m^3/s)', fontsize=11, labelpad=8)
    ax.set_ylabel('Head, H (m)', fontsize=11, labelpad=8)
    ax.set_zlabel('Absolute Error', fontsize=11, labelpad=8)
    ax.set_title(f'Prediction Error\nMean: {np.nanmean(E_error):.4f}', fontsize=12, pad=15)
    ax.view_init(elev=18, azim=-50, roll=0)
    ax.dist = 11
    
    plt.tight_layout()
    return fig

def create_ground_truth_2d(df, metric='Overall Eff', figsize=(6, 5)):
    """Ground truth 2D contour"""
    discharge = df['Dischargem'].values
    head = df['Head'].values
    efficiency = df[metric].values
    
    bep_idx = efficiency.argmax()
    
    discharge_grid = np.linspace(discharge.min(), discharge.max(), 150)
    head_grid = np.linspace(head.min(), head.max(), 150)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    E_grid = griddata((discharge, head), efficiency, (Q_grid, H_grid), method='cubic')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    levels = np.linspace(np.nanmin(E_grid), np.nanmax(E_grid), 25)
    cf = ax.contourf(Q_grid, H_grid, E_grid, levels=levels, cmap='RdYlGn', alpha=0.95)
    
    ax.scatter(discharge, head, c='blue', marker='o', s=30,
              edgecolors='darkblue', linewidths=0.8, alpha=0.7,
              label='Experimental data', zorder=5)
    
    ax.scatter([discharge[bep_idx]], [head[bep_idx]], c='red', marker='*',
              s=300, edgecolors='darkred', linewidths=2, label='BEP', zorder=10)
    
    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('Efficiency', fontsize=11)
    
    ax.set_xlabel('Discharge, Q (m^3/s)', fontsize=11)
    ax.set_ylabel('Head, H (m)', fontsize=11)
    ax.set_title(f'Ground Truth ({len(df)} exp)', fontsize=12, pad=10)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def create_bayesian_2d(df, sampled_indices, y_pred, bep_ground_truth, metric='Overall Eff', figsize=(6, 5)):
    """Bayesian prediction 2D contour"""
    discharge = df['Dischargem'].values
    head = df['Head'].values
    
    discharge_sampled = discharge[sampled_indices]
    head_sampled = head[sampled_indices]
    y_sampled = y_pred[sampled_indices]
    
    discharge_grid = np.linspace(discharge.min(), discharge.max(), 150)
    head_grid = np.linspace(head.min(), head.max(), 150)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    E_grid = griddata((discharge_sampled, head_sampled), y_sampled, (Q_grid, H_grid), method='cubic')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    levels = np.linspace(np.nanmin(E_grid), np.nanmax(E_grid), 25)
    cf = ax.contourf(Q_grid, H_grid, E_grid, levels=levels, cmap='RdYlGn', alpha=0.95)
    
    ax.scatter(discharge_sampled, head_sampled, c='red', marker='o', s=50,
              edgecolors='darkred', linewidths=1.2, alpha=0.9,
              label='Sampled experiments', zorder=5)
    
    ax.scatter([bep_ground_truth[0]], [bep_ground_truth[1]], c='gold', marker='*',
              s=300, edgecolors='black', linewidths=2,
              label='BEP (ground truth)', zorder=10)
    
    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('Efficiency', fontsize=11)
    
    ax.set_xlabel('Discharge, Q (m^3/s)', fontsize=11)
    ax.set_ylabel('Head, H (m)', fontsize=11)
    ax.set_title(f'Bayesian Prediction ({len(sampled_indices)} exp)', fontsize=12, pad=10)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def create_error_2d(df, sampled_indices, y_pred, metric='Overall Eff', figsize=(6, 5)):
    """Error 2D contour"""
    discharge = df['Dischargem'].values
    head = df['Head'].values
    y_true = df[metric].values
    
    discharge_grid = np.linspace(discharge.min(), discharge.max(), 150)
    head_grid = np.linspace(head.min(), head.max(), 150)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    
    E_true = griddata((discharge, head), y_true, (Q_grid, H_grid), method='cubic')
    
    discharge_sampled = discharge[sampled_indices]
    head_sampled = head[sampled_indices]
    y_sampled = y_pred[sampled_indices]
    E_pred = griddata((discharge_sampled, head_sampled), y_sampled, (Q_grid, H_grid), method='cubic')
    
    E_error = np.abs(E_true - E_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cf = ax.contourf(Q_grid, H_grid, E_error, levels=20, cmap='Reds', alpha=0.95)
    
    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('Absolute Error', fontsize=11)
    
    ax.set_xlabel('Discharge, Q (m^3/s)', fontsize=11)
    ax.set_ylabel('Head, H (m)', fontsize=11)
    ax.set_title(f'Prediction Error (Mean: {np.nanmean(E_error):.4f})', fontsize=12, pad=10)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def create_method_comparison_table(method_results, figsize=(10, 4)):
    """Method comparison table"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    methods = ['Bayesian', 'Random', 'Grid', 'LHS']
    
    table_data = [['Method', 'MAE', 'RMSE', 'R² Score', 'MAPE (%)']]
    
    for method in methods:
        results = method_results[method]
        row = [
            method,
            f"{results['mae']:.5f}",
            f"{results['rmse']:.5f}",
            f"{results['r2']:.5f}",
            f"{results['mape']:.2f}"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Find and highlight best values
    mae_values = [method_results[m]['mae'] for m in methods]
    r2_values = [method_results[m]['r2'] for m in methods]
    
    best_mae_idx = mae_values.index(min(mae_values))
    best_r2_idx = r2_values.index(max(r2_values))
    
    table[(best_mae_idx + 1, 1)].set_facecolor('#90EE90')
    table[(best_mae_idx + 1, 1)].set_text_props(weight='bold')
    
    table[(best_r2_idx + 1, 3)].set_facecolor('#90EE90')
    table[(best_r2_idx + 1, 3)].set_text_props(weight='bold')
    
    # Alternate rows
    for i in range(1, 5):
        for j in range(5):
            if i % 2 == 0:
                if table[(i, j)].get_facecolor() == (1.0, 1.0, 1.0, 1.0):
                    table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Sampling Methods Comparison\n(Best values highlighted)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig
    """BEP comparison for all methods"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    methods = ['Bayesian', 'Random', 'Grid', 'LHS']
    colors = ['lime', 'red', 'blue', 'orange']
    
    # Left: BEP locations
    ax1 = axes[0]
    ax1.scatter(bep_ground_truth[0], bep_ground_truth[1], c='gold', marker='*', s=800,
               edgecolors='black', linewidths=3, label='Ground Truth', zorder=10)
    
    for method, color in zip(methods, colors):
        bep = method_results[method]['bep']
        ax1.scatter(bep['discharge'], bep['head'], c=color, marker='o', s=300,
                   edgecolors='black', linewidths=2, label=method, zorder=5, alpha=0.8)
    
    ax1.set_xlabel('Discharge, Q (m^3/s)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Head, H (m)', fontsize=11, fontweight='bold')
    ax1.set_title('BEP Location Comparison', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(fontsize=9, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right: BEP errors
    ax2 = axes[1]
    discharge_errors = [method_results[m]['bep_error']['discharge'] for m in methods]
    head_errors = [method_results[m]['bep_error']['head'] for m in methods]
    eff_errors = [method_results[m]['bep_error']['efficiency'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax2.bar(x - width, discharge_errors, width, label='Discharge Error',
           color='steelblue', edgecolor='black', linewidth=1)
    ax2.bar(x, head_errors, width, label='Head Error',
           color='coral', edgecolor='black', linewidth=1)
    ax2.bar(x + width, eff_errors, width, label='Efficiency Error',
           color='lightgreen', edgecolor='black', linewidth=1)
    
    ax2.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
    ax2.set_title('BEP Prediction Error by Method', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.legend(fontsize=9, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    return fig

def main():
    # Create result directory
    result_dir = Path('result')
    result_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {result_dir.absolute()}")
    
    # Load data
    print("\nLoading ground truth data from bf-gv folder...")
    df = load_all_data('bf-gv')
    print(f"Loaded {len(df)} experiments")
    
    # Parameters
    n_initial = 5
    n_iterations = 10
    n_samples = n_initial + n_iterations
    
    print(f"\n{'='*70}")
    print(f"COMPARING SAMPLING METHODS ({n_samples} samples each)")
    print(f"{'='*70}")
    
    # Run all methods
    print("\n1. Bayesian Optimization...")
    bayesian_indices, bayesian_pred, bayesian_std = simulate_bayesian_sampling(
        df, n_initial, n_iterations, 'Overall Eff')
    bayesian_metrics = calculate_metrics(df, bayesian_pred, 'Overall Eff')
    
    print("\n2. Random Sampling...")
    random_indices, random_pred, random_std = simulate_random_sampling(df, n_samples, 'Overall Eff')
    random_metrics = calculate_metrics(df, random_pred, 'Overall Eff')
    
    print("\n3. Grid Sampling...")
    grid_indices, grid_pred, grid_std = simulate_grid_sampling(df, n_samples, 'Overall Eff')
    grid_metrics = calculate_metrics(df, grid_pred, 'Overall Eff')
    
    print("\n4. Latin Hypercube Sampling...")
    lhs_indices, lhs_pred, lhs_std = simulate_lhs_sampling(df, n_samples, 'Overall Eff')
    lhs_metrics = calculate_metrics(df, lhs_pred, 'Overall Eff')
    
    # Compile results
    method_results = {
        'Bayesian': {'indices': bayesian_indices, 'predictions': bayesian_pred, **bayesian_metrics},
        'Random': {'indices': random_indices, 'predictions': random_pred, **random_metrics},
        'Grid': {'indices': grid_indices, 'predictions': grid_pred, **grid_metrics},
        'LHS': {'indices': lhs_indices, 'predictions': lhs_pred, **lhs_metrics}
    }
    
    # Find BEP for each method
    for method in ['Bayesian', 'Random', 'Grid', 'LHS']:
        bep = find_bep(df, method_results[method]['indices'], method_results[method]['predictions'])
        method_results[method]['bep'] = bep
    
    # Ground truth BEP
    y_true = df['Overall Eff'].values
    true_bep_idx = y_true.argmax()
    bep_ground_truth = (
        df.iloc[true_bep_idx]['Dischargem'],
        df.iloc[true_bep_idx]['Head'],
        y_true[true_bep_idx],
        df.iloc[true_bep_idx]['G/V degree']
    )
    
    # Calculate BEP errors
    for method in ['Bayesian', 'Random', 'Grid', 'LHS']:
        bep = method_results[method]['bep']
        method_results[method]['bep_error'] = {
            'discharge': abs(bep['discharge'] - bep_ground_truth[0]),
            'head': abs(bep['head'] - bep_ground_truth[1]),
            'efficiency': abs(bep['efficiency'] - bep_ground_truth[2])
        }
    
    print(f"\n{'='*70}")
    print("Generating figures...")
    print(f"{'='*70}")
    
    # Figures 1-6: Bayesian Optimization Details
    fig1, bep_ground_truth = create_ground_truth_3d(df, 'Overall Eff')
    fig1.savefig(result_dir / 'fig1_ground_truth_3d.png', dpi=600, bbox_inches='tight')
    fig1.savefig(result_dir / 'fig1_ground_truth_3d.pdf', bbox_inches='tight')
    print("✓ fig1_ground_truth_3d")
    
    fig2 = create_bayesian_3d(df, bayesian_indices, bayesian_pred, bep_ground_truth, 'Overall Eff')
    fig2.savefig(result_dir / 'fig2_bayesian_3d.png', dpi=600, bbox_inches='tight')
    fig2.savefig(result_dir / 'fig2_bayesian_3d.pdf', bbox_inches='tight')
    print("✓ fig2_bayesian_3d")
    
    fig3 = create_error_3d(df, bayesian_indices, bayesian_pred, 'Overall Eff')
    fig3.savefig(result_dir / 'fig3_error_3d.png', dpi=600, bbox_inches='tight')
    fig3.savefig(result_dir / 'fig3_error_3d.pdf', bbox_inches='tight')
    print("✓ fig3_error_3d")
    
    fig4 = create_ground_truth_2d(df, 'Overall Eff')
    fig4.savefig(result_dir / 'fig4_ground_truth_2d.png', dpi=600, bbox_inches='tight')
    fig4.savefig(result_dir / 'fig4_ground_truth_2d.pdf', bbox_inches='tight')
    print("✓ fig4_ground_truth_2d")
    
    fig5 = create_bayesian_2d(df, bayesian_indices, bayesian_pred, bep_ground_truth, 'Overall Eff')
    fig5.savefig(result_dir / 'fig5_bayesian_2d.png', dpi=600, bbox_inches='tight')
    fig5.savefig(result_dir / 'fig5_bayesian_2d.pdf', bbox_inches='tight')
    print("✓ fig5_bayesian_2d")
    
    fig6 = create_error_2d(df, bayesian_indices, bayesian_pred, 'Overall Eff')
    fig6.savefig(result_dir / 'fig6_error_2d.png', dpi=600, bbox_inches='tight')
    fig6.savefig(result_dir / 'fig6_error_2d.pdf', bbox_inches='tight')
    print("✓ fig6_error_2d")
    
    # Figures 7-11: Method Comparison
    fig7 = create_method_comparison_surfaces(df, method_results, 'Overall Eff')
    fig7.savefig(result_dir / 'fig7_method_comparison_surfaces.png', dpi=600, bbox_inches='tight')
    fig7.savefig(result_dir / 'fig7_method_comparison_surfaces.pdf', bbox_inches='tight')
    print("✓ fig7_method_comparison_surfaces")
    
    fig8 = create_bep_comparison_all_methods(bep_ground_truth, method_results)
    fig8.savefig(result_dir / 'fig8_bep_comparison_all_methods.png', dpi=600, bbox_inches='tight')
    fig8.savefig(result_dir / 'fig8_bep_comparison_all_methods.pdf', bbox_inches='tight')
    print("✓ fig8_bep_comparison_all_methods")
    
    fig9 = create_bep_comparison_table_all_methods(bep_ground_truth, method_results)
    fig9.savefig(result_dir / 'fig9_bep_comparison_table.png', dpi=600, bbox_inches='tight')
    fig9.savefig(result_dir / 'fig9_bep_comparison_table.pdf', bbox_inches='tight')
    print("✓ fig9_bep_comparison_table")
    
    fig10 = create_method_comparison_metrics(method_results)
    fig10.savefig(result_dir / 'fig10_method_comparison_metrics.png', dpi=600, bbox_inches='tight')
    fig10.savefig(result_dir / 'fig10_method_comparison_metrics.pdf', bbox_inches='tight')
    print("✓ fig10_method_comparison_metrics")
    
    fig11 = create_method_comparison_table(method_results)
    fig11.savefig(result_dir / 'fig11_method_comparison_table.png', dpi=600, bbox_inches='tight')
    fig11.savefig(result_dir / 'fig11_method_comparison_table.pdf', bbox_inches='tight')
    print("✓ fig11_method_comparison_table")
    
    # Save data
    print(f"\nSaving data files...")
    for method_name, method_data in method_results.items():
        method_df = df.iloc[method_data['indices']].copy()
        method_df['Predicted_Eff'] = method_data['predictions'][method_data['indices']]
        method_df.to_csv(result_dir / f'{method_name.lower()}_sampled_points.csv', index=False)
    print("✓ Sampled points for all methods")
    
    # Summary
    summary = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(df),
            'samples_per_method': n_samples,
            'reduction_percentage': (1 - n_samples/len(df)) * 100
        },
        'bep_ground_truth': {
            'discharge': float(bep_ground_truth[0]),
            'head': float(bep_ground_truth[1]),
            'efficiency': float(bep_ground_truth[2])
        },
        'methods': {}
    }
    
    for method in ['Bayesian', 'Random', 'Grid', 'LHS']:
        data = method_results[method]
        summary['methods'][method] = {
            'mae': float(data['mae']),
            'rmse': float(data['rmse']),
            'r2': float(data['r2']),
            'mape': float(data['mape']),
            'bep': {
                'discharge': float(data['bep']['discharge']),
                'head': float(data['bep']['head']),
                'efficiency': float(data['bep']['efficiency'])
            },
            'bep_error': {
                'discharge': float(data['bep_error']['discharge']),
                'head': float(data['bep_error']['head']),
                'efficiency': float(data['bep_error']['efficiency'])
            }
        }
    
    with open(result_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("✓ summary.json")
    
    # Print results
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nExperiment Reduction: {len(df)} → {n_samples} ({(1-n_samples/len(df))*100:.1f}%)")
    print(f"\nMethod Comparison:")
    print(f"{'Method':<12} {'MAE':<10} {'R²':<10} {'BEP Error':<12}")
    print("-" * 44)
    
    for method in ['Bayesian', 'Random', 'Grid', 'LHS']:
        data = method_results[method]
        bep_total = sum(data['bep_error'].values())
        print(f"{method:<12} {data['mae']:<10.5f} {data['r2']:<10.4f} {bep_total:<12.5f}")
    
    improvement = ((random_metrics['mae'] - bayesian_metrics['mae']) / random_metrics['mae']) * 100
    print(f"\nBayesian improvement over Random: {improvement:.1f}%")
    print(f"\n{'='*70}")
    print(f"All results saved to: {result_dir.absolute()}")
    print(f"{'='*70}")
    
    plt.show()

if __name__ == "__main__":
    main()