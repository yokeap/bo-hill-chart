"""
Visualization functions for hill chart analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata


def _overlay_gv_lines_3d(ax, Q_grid, H_grid, E_grid, GV_grid, gv):
    """Draw G/V angle iso-lines on a 3D efficiency surface"""
    gv_levels = sorted(np.unique(gv))

    # Extract 2D contour paths using a temporary figure
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(Q_grid, H_grid, GV_grid, levels=gv_levels)
    plt.close(fig_tmp)

    q_flat = Q_grid.ravel()
    h_flat = H_grid.ravel()
    e_flat = E_grid.ravel()

    for i, segs in enumerate(cs.allsegs):
        gv_val = gv_levels[i]
        labeled = False
        for seg in segs:
            if len(seg) < 2:
                continue
            q_line = seg[:, 0]
            h_line = seg[:, 1]
            z_line = griddata((q_flat, h_flat), e_flat,
                              (q_line, h_line), method='linear')
            valid = ~np.isnan(z_line)
            if valid.sum() < 2:
                continue
            ax.plot(q_line[valid], h_line[valid], z_line[valid],
                   color='white', linewidth=0.9, linestyle='--', alpha=0.85, zorder=5)
            if not labeled:
                mid = np.where(valid)[0][valid.sum() // 2]
                ax.text(q_line[mid], h_line[mid], z_line[mid],
                       f' {gv_val:.0f}°', fontsize=7, color='white',
                       fontweight='bold', zorder=6)
                labeled = True


def create_ground_truth_3d(df, metric='Overall Eff', figsize=(8, 6)):
    """Ground truth 3D surface with G/V angle iso-lines on the surface"""
    discharge = df['Dischargem'].values
    head = df['Head'].values
    efficiency = df[metric].values
    gv = df['G/V degree'].values

    bep_idx = efficiency.argmax()
    bep_info = (discharge[bep_idx], head[bep_idx], efficiency[bep_idx],
                df.iloc[bep_idx]['G/V degree'])

    discharge_grid = np.linspace(discharge.min(), discharge.max(), 100)
    head_grid = np.linspace(head.min(), head.max(), 100)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    E_grid = griddata((discharge, head), efficiency, (Q_grid, H_grid), method='cubic')
    GV_grid = griddata((discharge, head), gv, (Q_grid, H_grid), method='linear')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(Q_grid, H_grid, E_grid, cmap='RdYlGn', alpha=0.80,
                          edgecolor='none', antialiased=True, shade=True)
    _overlay_gv_lines_3d(ax, Q_grid, H_grid, E_grid, GV_grid, gv)

    ax.scatter(discharge, head, efficiency, c='blue', marker='o', s=30,
              edgecolors='darkblue', linewidths=0.4, alpha=0.85,
              label='Experimental data')
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


def create_bayesian_3d(df, sampled_indices, y_pred, bep_ground_truth, 
                       metric='Overall Eff', figsize=(8, 6)):
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
    
    discharge_grid = np.linspace(discharge.min(), discharge.max(), 100)
    head_grid = np.linspace(head.min(), head.max(), 100)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    E_grid = griddata((discharge_sampled, head_sampled), y_sampled, 
                     (Q_grid, H_grid), method='cubic')
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(Q_grid, H_grid, E_grid, cmap='RdYlGn', alpha=0.85,
                          edgecolor='none', antialiased=True, shade=True)
    
    ax.scatter(discharge_sampled, head_sampled, y_sampled, c='red', marker='o', s=60,
              edgecolors='darkred', linewidths=1.5, alpha=0.9, label='Sampled experiments')
    
    ax.scatter([bep_ground_truth[0]], [bep_ground_truth[1]], [bep_ground_truth[2]],
              c='gold', marker='*', s=400, edgecolors='black', linewidths=2,
              label='BEP (ground truth)', zorder=10)
    
    ax.scatter([bayesian_bep['discharge']], [bayesian_bep['head']], 
              [bayesian_bep['efficiency']], c='lime', marker='*', s=400, 
              edgecolors='black', linewidths=2, label='BEP (Bayesian)', zorder=10)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.08)
    cbar.set_label('Efficiency', fontsize=11)
    
    ax.set_xlabel('Discharge, Q (m^3/s)', fontsize=11, labelpad=8)
    ax.set_ylabel('Head, H (m)', fontsize=11, labelpad=8)
    ax.set_zlabel('Efficiency', fontsize=11, labelpad=8)
    ax.set_title(f'Bayesian Prediction\n({len(sampled_indices)} experiments)', 
                fontsize=12, pad=15)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.view_init(elev=18, azim=-50, roll=0)
    ax.dist = 11
    
    plt.tight_layout()
    return fig


def create_error_3d(df, sampled_indices, y_pred, metric='Overall Eff',
                    method_name='', figsize=(8, 6)):
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
    E_pred = griddata((discharge_sampled, head_sampled), y_sampled,
                     (Q_grid, H_grid), method='cubic')

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

    if method_name:
        title = f'Abs. Error: Ground Truth vs {method_name}\n(Mean: {np.nanmean(E_error):.4f})'
    else:
        title = f'Prediction Error\nMean: {np.nanmean(E_error):.4f}'
    ax.set_title(title, fontsize=12, pad=15)

    ax.view_init(elev=18, azim=-50, roll=0)
    ax.dist = 11

    plt.tight_layout()
    return fig


def create_ground_truth_2d(df, metric='Overall Eff', figsize=(6, 5)):
    """Ground truth 2D contour with guide vane angle iso-lines"""
    discharge = df['Dischargem'].values
    head = df['Head'].values
    efficiency = df[metric].values
    gv = df['G/V degree'].values

    bep_idx = efficiency.argmax()

    discharge_grid = np.linspace(discharge.min(), discharge.max(), 150)
    head_grid = np.linspace(head.min(), head.max(), 150)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    E_grid = griddata((discharge, head), efficiency, (Q_grid, H_grid), method='cubic')
    GV_grid = griddata((discharge, head), gv, (Q_grid, H_grid), method='linear')

    fig, ax = plt.subplots(figsize=figsize)

    levels = np.linspace(np.nanmin(E_grid), np.nanmax(E_grid), 25)
    cf = ax.contourf(Q_grid, H_grid, E_grid, levels=levels, cmap='RdYlGn', alpha=0.95)

    # Guide vane angle iso-lines
    gv_levels = sorted(np.unique(gv))
    cs = ax.contour(Q_grid, H_grid, GV_grid, levels=gv_levels,
                   colors='white', linewidths=0.9, linestyles='--', alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f°')

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


def create_bayesian_2d(df, sampled_indices, y_pred, bep_ground_truth, 
                       metric='Overall Eff', figsize=(6, 5)):
    """Bayesian prediction 2D contour"""
    discharge = df['Dischargem'].values
    head = df['Head'].values
    
    discharge_sampled = discharge[sampled_indices]
    head_sampled = head[sampled_indices]
    y_sampled = y_pred[sampled_indices]
    
    discharge_grid = np.linspace(discharge.min(), discharge.max(), 150)
    head_grid = np.linspace(head.min(), head.max(), 150)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    E_grid = griddata((discharge_sampled, head_sampled), y_sampled, 
                     (Q_grid, H_grid), method='cubic')
    
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


def create_error_2d(df, sampled_indices, y_pred, metric='Overall Eff',
                    method_name='', figsize=(6, 5)):
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
    E_pred = griddata((discharge_sampled, head_sampled), y_sampled,
                     (Q_grid, H_grid), method='cubic')

    E_error = np.abs(E_true - E_pred)

    fig, ax = plt.subplots(figsize=figsize)

    cf = ax.contourf(Q_grid, H_grid, E_error, levels=20, cmap='Reds', alpha=0.95)

    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('Absolute Error', fontsize=11)

    ax.set_xlabel('Discharge, Q (m^3/s)', fontsize=11)
    ax.set_ylabel('Head, H (m)', fontsize=11)

    if method_name:
        title = f'Abs. Error: Ground Truth vs {method_name}\n(Mean: {np.nanmean(E_error):.4f})'
    else:
        title = f'Prediction Error (Mean: {np.nanmean(E_error):.4f})'
    ax.set_title(title, fontsize=12, pad=10)

    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return fig


def create_method_comparison_surfaces(df, method_results, metric='Overall Eff', 
                                      figsize=(15, 10)):
    """Compare all sampling methods"""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    discharge = df['Dischargem'].values
    head = df['Head'].values
    
    methods = ['Bayesian', 'Random', 'Grid', 'LHS', 'Fibonacci']
    colors = ['lime', 'red', 'blue', 'orange', 'magenta']
    
    for idx, (method, color) in enumerate(zip(methods, colors)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
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
    
    # Hide the 6th subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


def create_method_comparison_metrics(method_results, figsize=(10, 6)):
    """Bar chart comparing method performance"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    methods = ['Bayesian', 'Random', 'Grid', 'LHS', 'Fibonacci']
    colors = ['lime', 'red', 'blue', 'orange', 'magenta']
    
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


def create_bep_comparison_all_methods(bep_ground_truth, method_results, figsize=(14, 5)):
    """BEP comparison for all methods"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    methods = ['Bayesian', 'Random', 'Grid', 'LHS', 'Fibonacci']
    colors = ['lime', 'red', 'blue', 'orange', 'magenta']
    
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


def create_bep_comparison_table_all_methods(bep_ground_truth, method_results, 
                                            figsize=(13, 4)):
    """Table showing BEP comparison for all methods"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    methods = ['Ground Truth', 'Bayesian', 'Random', 'Grid', 'LHS', 'Fibonacci']
    
    table_data = [['Method', 'Discharge (m^3/s)', 'Head (m)', 'Efficiency', 
                   'Q Error', 'H Error', 'Eff Error']]
    
    # Ground truth row
    table_data.append([
        'Ground Truth',
        f"{bep_ground_truth[0]:.4f}",
        f"{bep_ground_truth[1]:.2f}",
        f"{bep_ground_truth[2]:.4f}",
        '-', '-', '-'
    ])
    
    # Each method row
    for method in ['Bayesian', 'Random', 'Grid', 'LHS', 'Fibonacci']:
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
                    colWidths=[0.15, 0.14, 0.11, 0.13, 0.11, 0.11, 0.11])
    
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
    
    # Find best method
    total_errors = []
    for method in ['Bayesian', 'Random', 'Grid', 'LHS', 'Fibonacci']:
        error = method_results[method]['bep_error']
        total = error['discharge'] + error['head'] + error['efficiency']
        total_errors.append(total)
    
    best_method_idx = total_errors.index(min(total_errors)) + 2
    
    # Highlight best method
    for i in range(7):
        table[(best_method_idx, i)].set_facecolor('#90EE90')
        table[(best_method_idx, i)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(2, 7):
        if i != best_method_idx:
            for j in range(7):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('BEP Comparison: All Methods\n(Best method highlighted in green)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def create_method_comparison_table(method_results, figsize=(11, 4)):
    """Method comparison table"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    methods = ['Bayesian', 'Random', 'Grid', 'LHS', 'Fibonacci']
    
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
    for i in range(1, 6):
        for j in range(5):
            if i % 2 == 0:
                if table[(i, j)].get_facecolor() == (1.0, 1.0, 1.0, 1.0):
                    table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Sampling Methods Comparison\n(Best values highlighted)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def create_method_3d(df, sampled_indices, y_pred, bep_ground_truth,
                     method_name, sample_color, metric='Overall Eff', figsize=(8, 6)):
    """Generic prediction 3D surface for any sampling method, with G/V angle iso-lines"""
    discharge = df['Dischargem'].values
    head = df['Head'].values
    gv = df['G/V degree'].values

    discharge_sampled = discharge[sampled_indices]
    head_sampled = head[sampled_indices]
    y_sampled = y_pred[sampled_indices]

    bep_idx = np.argmax(y_sampled)
    method_bep = {
        'discharge': discharge_sampled[bep_idx],
        'head': head_sampled[bep_idx],
        'efficiency': y_sampled[bep_idx],
    }

    discharge_grid = np.linspace(discharge.min(), discharge.max(), 100)
    head_grid = np.linspace(head.min(), head.max(), 100)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    E_grid = griddata((discharge_sampled, head_sampled), y_sampled,
                     (Q_grid, H_grid), method='cubic')
    GV_grid = griddata((discharge, head), gv, (Q_grid, H_grid), method='linear')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(Q_grid, H_grid, E_grid, cmap='RdYlGn', alpha=0.80,
                          edgecolor='none', antialiased=True, shade=True)
    _overlay_gv_lines_3d(ax, Q_grid, H_grid, E_grid, GV_grid, gv)

    ax.scatter(discharge_sampled, head_sampled, y_sampled, c=sample_color, marker='o', s=60,
              edgecolors='black', linewidths=1, alpha=0.9,
              label=f'{method_name} samples')
    ax.scatter([bep_ground_truth[0]], [bep_ground_truth[1]], [bep_ground_truth[2]],
              c='gold', marker='*', s=400, edgecolors='black', linewidths=2,
              label='BEP (ground truth)', zorder=10)
    ax.scatter([method_bep['discharge']], [method_bep['head']],
              [method_bep['efficiency']], c='lime', marker='*', s=400,
              edgecolors='black', linewidths=2, label=f'BEP ({method_name})', zorder=10)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.08)
    cbar.set_label('Efficiency', fontsize=11)

    ax.set_xlabel('Discharge, Q (m^3/s)', fontsize=11, labelpad=8)
    ax.set_ylabel('Head, H (m)', fontsize=11, labelpad=8)
    ax.set_zlabel('Efficiency', fontsize=11, labelpad=8)
    ax.set_title(f'{method_name} Prediction\n({len(sampled_indices)} experiments)',
                fontsize=12, pad=15)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.view_init(elev=18, azim=-50, roll=0)
    ax.dist = 11

    plt.tight_layout()
    return fig


def create_method_2d(df, sampled_indices, y_pred, bep_ground_truth,
                     method_name, sample_color, metric='Overall Eff', figsize=(6, 5)):
    """Generic prediction 2D contour for any sampling method, with G/V angle iso-lines"""
    discharge = df['Dischargem'].values
    head = df['Head'].values
    gv = df['G/V degree'].values

    discharge_sampled = discharge[sampled_indices]
    head_sampled = head[sampled_indices]
    y_sampled = y_pred[sampled_indices]

    discharge_grid = np.linspace(discharge.min(), discharge.max(), 150)
    head_grid = np.linspace(head.min(), head.max(), 150)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    E_grid = griddata((discharge_sampled, head_sampled), y_sampled,
                     (Q_grid, H_grid), method='cubic')
    GV_grid = griddata((discharge, head), gv, (Q_grid, H_grid), method='linear')

    fig, ax = plt.subplots(figsize=figsize)

    levels = np.linspace(np.nanmin(E_grid), np.nanmax(E_grid), 25)
    cf = ax.contourf(Q_grid, H_grid, E_grid, levels=levels, cmap='RdYlGn', alpha=0.95)

    gv_levels = sorted(np.unique(gv))
    cs = ax.contour(Q_grid, H_grid, GV_grid, levels=gv_levels,
                   colors='white', linewidths=0.9, linestyles='--', alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f°')

    ax.scatter(discharge_sampled, head_sampled, c=sample_color, marker='o', s=50,
              edgecolors='black', linewidths=1.2, alpha=0.9,
              label=f'{method_name} samples', zorder=5)
    ax.scatter([bep_ground_truth[0]], [bep_ground_truth[1]], c='gold', marker='*',
              s=300, edgecolors='black', linewidths=2,
              label='BEP (ground truth)', zorder=10)

    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('Efficiency', fontsize=11)

    ax.set_xlabel('Discharge, Q (m^3/s)', fontsize=11)
    ax.set_ylabel('Head, H (m)', fontsize=11)
    ax.set_title(f'{method_name} Prediction ({len(sampled_indices)} exp)', fontsize=12, pad=10)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return fig


def create_fibonacci_3d(df, sampled_indices, y_pred, bep_ground_truth,
                        metric='Overall Eff', figsize=(8, 6)):
    """Golden Ratio (Fibonacci) prediction 3D surface with G/V angle iso-lines"""
    discharge = df['Dischargem'].values
    head = df['Head'].values
    gv = df['G/V degree'].values

    discharge_sampled = discharge[sampled_indices]
    head_sampled = head[sampled_indices]
    y_sampled = y_pred[sampled_indices]

    bep_idx = np.argmax(y_sampled)
    fibonacci_bep = {
        'discharge': discharge_sampled[bep_idx],
        'head': head_sampled[bep_idx],
        'efficiency': y_sampled[bep_idx],
    }

    discharge_grid = np.linspace(discharge.min(), discharge.max(), 100)
    head_grid = np.linspace(head.min(), head.max(), 100)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    E_grid = griddata((discharge_sampled, head_sampled), y_sampled,
                     (Q_grid, H_grid), method='cubic')
    GV_grid = griddata((discharge, head), gv, (Q_grid, H_grid), method='linear')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(Q_grid, H_grid, E_grid, cmap='RdYlGn', alpha=0.80,
                          edgecolor='none', antialiased=True, shade=True)
    _overlay_gv_lines_3d(ax, Q_grid, H_grid, E_grid, GV_grid, gv)

    ax.scatter(discharge_sampled, head_sampled, y_sampled, c='magenta', marker='o', s=60,
              edgecolors='darkmagenta', linewidths=1, alpha=0.9,
              label='Golden Ratio samples')
    ax.scatter([bep_ground_truth[0]], [bep_ground_truth[1]], [bep_ground_truth[2]],
              c='gold', marker='*', s=400, edgecolors='black', linewidths=2,
              label='BEP (ground truth)', zorder=10)
    ax.scatter([fibonacci_bep['discharge']], [fibonacci_bep['head']],
              [fibonacci_bep['efficiency']], c='lime', marker='*', s=400,
              edgecolors='black', linewidths=2, label='BEP (Golden Ratio)', zorder=10)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.08)
    cbar.set_label('Efficiency', fontsize=11)

    ax.set_xlabel('Discharge, Q (m^3/s)', fontsize=11, labelpad=8)
    ax.set_ylabel('Head, H (m)', fontsize=11, labelpad=8)
    ax.set_zlabel('Efficiency', fontsize=11, labelpad=8)
    ax.set_title(f'Golden Ratio (Fibonacci Spiral) Prediction\n({len(sampled_indices)} experiments)',
                fontsize=12, pad=15)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.view_init(elev=18, azim=-50, roll=0)
    ax.dist = 11

    plt.tight_layout()
    return fig


def create_fibonacci_2d(df, sampled_indices, y_pred, bep_ground_truth,
                        metric='Overall Eff', figsize=(6, 5)):
    """Golden Ratio (Fibonacci) prediction 2D contour with G/V angle iso-lines"""
    discharge = df['Dischargem'].values
    head = df['Head'].values
    gv = df['G/V degree'].values

    discharge_sampled = discharge[sampled_indices]
    head_sampled = head[sampled_indices]
    y_sampled = y_pred[sampled_indices]

    discharge_grid = np.linspace(discharge.min(), discharge.max(), 150)
    head_grid = np.linspace(head.min(), head.max(), 150)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    E_grid = griddata((discharge_sampled, head_sampled), y_sampled,
                     (Q_grid, H_grid), method='cubic')
    GV_grid = griddata((discharge, head), gv, (Q_grid, H_grid), method='linear')

    fig, ax = plt.subplots(figsize=figsize)

    levels = np.linspace(np.nanmin(E_grid), np.nanmax(E_grid), 25)
    cf = ax.contourf(Q_grid, H_grid, E_grid, levels=levels, cmap='RdYlGn', alpha=0.95)

    gv_levels = sorted(np.unique(gv))
    cs = ax.contour(Q_grid, H_grid, GV_grid, levels=gv_levels,
                   colors='white', linewidths=0.9, linestyles='--', alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f°')

    ax.scatter(discharge_sampled, head_sampled, c='magenta', marker='o', s=50,
              edgecolors='darkmagenta', linewidths=1.2, alpha=0.9,
              label='Golden Ratio samples', zorder=5)
    ax.scatter([bep_ground_truth[0]], [bep_ground_truth[1]], c='gold', marker='*',
              s=300, edgecolors='black', linewidths=2,
              label='BEP (ground truth)', zorder=10)

    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('Efficiency', fontsize=11)

    ax.set_xlabel('Discharge, Q (m^3/s)', fontsize=11)
    ax.set_ylabel('Head, H (m)', fontsize=11)
    ax.set_title(f'Golden Ratio Prediction ({len(sampled_indices)} exp)', fontsize=12, pad=10)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return fig