import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm

def load_all_data(folder_path='bf-bulb'):
    """Load all CSV files and combine them with head information"""
    folder = Path(folder_path)
    all_data = []
    
    csv_files = sorted(folder.glob('*.csv'))
    
    for file in csv_files:
        # Extract head from filename (e.g., '10m.csv' -> 10)
        head = float(file.stem.replace('m', ''))
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Add head column
        df['Head'] = head
        
        # Filter out invalid rows (efficiency > 1 indicates placeholder data)
        df = df[(df['Overall Eff'] < 1) & (df['Dischargem'].notna())]
        
        all_data.append(df)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def create_traditional_hill_chart_3d(df, metric='Overall Eff', figsize=(16, 12)):
    """Create traditional 3D hill chart: Discharge vs Head vs Efficiency"""
    
    # Extract data
    discharge = df['Dischargem'].values
    head = df['Head'].values
    efficiency = df[metric].values
    gv_angle = df['G/V degree'].values
    
    # Find BEP (Best Efficiency Point)
    bep_idx = efficiency.argmax()
    bep_discharge = discharge[bep_idx]
    bep_head = head[bep_idx]
    bep_efficiency = efficiency[bep_idx]
    bep_gv = gv_angle[bep_idx]
    
    print(f"\nBest Efficiency Point (BEP):")
    print(f"  Discharge: {bep_discharge:.4f} m³/s")
    print(f"  Head: {bep_head:.1f} m")
    print(f"  Guide Vane Angle: {bep_gv:.1f}°")
    print(f"  {metric}: {bep_efficiency:.4f}")
    
    # Create interpolation grid for smooth surface
    discharge_grid = np.linspace(discharge.min(), discharge.max(), 80)
    head_grid = np.linspace(head.min(), head.max(), 80)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    
    # Interpolate efficiency values
    E_grid = griddata((discharge, head), efficiency, (Q_grid, H_grid), method='cubic')
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(Q_grid, H_grid, E_grid, cmap='RdYlGn', 
                          alpha=0.75, edgecolor='none', antialiased=True,
                          vmin=np.nanmin(E_grid), vmax=np.nanmax(E_grid))
    
    # Plot guide vane angle curves
    unique_gv = sorted(df['G/V degree'].unique())
    gv_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_gv)))
    
    for i, gv in enumerate(unique_gv):
        df_gv = df[df['G/V degree'] == gv].sort_values('Head')
        if len(df_gv) > 1:
            ax.plot(df_gv['Dischargem'], df_gv['Head'], 
                   df_gv[metric], color=gv_colors[i], 
                   linewidth=2.5, marker='o', markersize=5,
                   label=f'G/V {gv:.0f}°', alpha=0.9, zorder=5)
    
    # Highlight BEP
    ax.scatter([bep_discharge], [bep_head], [bep_efficiency], 
              c='red', marker='*', s=1000, edgecolors='black', 
              linewidths=3, label='BEP', zorder=10)
    
    # Add vertical line from BEP to base
    ax.plot([bep_discharge, bep_discharge], 
            [bep_head, bep_head], 
            [np.nanmin(E_grid)*0.95, bep_efficiency], 
            'r--', linewidth=3, alpha=0.8, zorder=9)
    
    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label(metric, fontsize=12, fontweight='bold')
    
    # Labels
    ax.set_xlabel('Discharge Q (m³/s)', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('Head H (m)', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_zlabel(metric + ' η', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_title(f'Traditional Hill Chart - 3D View\nBulb Turbine Performance Map', 
                fontsize=16, fontweight='bold', pad=25)
    
    # Legend
    ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.95)
    
    # Set viewing angle for better visualization
    ax.view_init(elev=25, azim=225)
    
    plt.tight_layout()
    return fig, ax

def create_traditional_hill_chart_contour(df, metric='Overall Eff', figsize=(14, 11)):
    """Create traditional 2D hill chart: Discharge vs Head with efficiency contours"""
    
    # Extract data
    discharge = df['Dischargem'].values
    head = df['Head'].values
    efficiency = df[metric].values
    
    # Find BEP
    bep_idx = efficiency.argmax()
    bep_discharge = discharge[bep_idx]
    bep_head = head[bep_idx]
    bep_efficiency = efficiency[bep_idx]
    bep_gv = df.iloc[bep_idx]['G/V degree']
    
    # Create interpolation grid
    discharge_grid = np.linspace(discharge.min(), discharge.max(), 150)
    head_grid = np.linspace(head.min(), head.max(), 150)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    
    # Interpolate efficiency values
    E_grid = griddata((discharge, head), efficiency, (Q_grid, H_grid), method='cubic')
    
    # Create 2D contour plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filled contours for efficiency
    levels = np.linspace(np.nanmin(E_grid), np.nanmax(E_grid), 30)
    contourf = ax.contourf(Q_grid, H_grid, E_grid, levels=levels, cmap='RdYlGn', alpha=0.95)
    contour = ax.contour(Q_grid, H_grid, E_grid, levels=levels[::2], colors='black', 
                        linewidths=0.5, alpha=0.4)
    
    # Add contour labels for efficiency
    ax.clabel(contour, inline=True, fontsize=9, fmt='%.3f')
    
    # Plot guide vane angle curves
    unique_gv = sorted(df['G/V degree'].unique())
    gv_colors = plt.cm.tab20b(np.linspace(0, 1, len(unique_gv)))
    
    for i, gv in enumerate(unique_gv):
        df_gv = df[df['G/V degree'] == gv].sort_values('Head')
        if len(df_gv) > 1:
            ax.plot(df_gv['Dischargem'], df_gv['Head'], 
                   color=gv_colors[i], linewidth=2.5, marker='o', 
                   markersize=7, markeredgecolor='white', markeredgewidth=1.5,
                   label=f'{gv:.0f}°', alpha=0.9, zorder=5)
    
    # Highlight BEP
    ax.scatter([bep_discharge], [bep_head], c='red', marker='*', 
              s=1200, edgecolors='black', linewidths=3, 
              label='BEP', zorder=10)
    
    # Add BEP annotation
    ax.annotate(f'BEP\nη = {bep_efficiency:.4f}\nQ = {bep_discharge:.3f} m³/s\nH = {bep_head:.1f} m\nG/V = {bep_gv:.1f}°',
               xy=(bep_discharge, bep_head), xytext=(40, 40),
               textcoords='offset points', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', 
                        edgecolor='red', linewidth=2.5, alpha=0.95),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                             lw=3, color='red'))
    
    # Colorbar
    cbar = plt.colorbar(contourf, ax=ax, pad=0.02)
    cbar.set_label(metric + ' (η)', fontsize=13, fontweight='bold')
    
    # Labels
    ax.set_xlabel('Discharge Q (m³/s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Head H (m)', fontsize=14, fontweight='bold')
    ax.set_title('Traditional Operating Hill Chart\nBulb Turbine Performance Map\n(Lines show constant Guide Vane angles)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Legend for guide vane angles
    legend = ax.legend(title='Guide Vane\nAngle', loc='best', fontsize=9, 
                      ncol=2, framealpha=0.95, title_fontsize=10)
    legend.get_title().set_fontweight('bold')
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    return fig, ax

def create_power_hill_chart(df, figsize=(14, 11)):
    """Create hill chart for power output"""
    
    # Extract data
    discharge = df['Dischargem'].values
    head = df['Head'].values
    power = df['Generator power'].values
    
    # Find maximum power point
    max_power_idx = power.argmax()
    max_discharge = discharge[max_power_idx]
    max_head = head[max_power_idx]
    max_power = power[max_power_idx]
    max_gv = df.iloc[max_power_idx]['G/V degree']
    
    print(f"\nMaximum Power Point:")
    print(f"  Discharge: {max_discharge:.4f} m³/s")
    print(f"  Head: {max_head:.1f} m")
    print(f"  Guide Vane Angle: {max_gv:.1f}°")
    print(f"  Generator Power: {max_power:.2f} kW")
    
    # Create interpolation grid
    discharge_grid = np.linspace(discharge.min(), discharge.max(), 150)
    head_grid = np.linspace(head.min(), head.max(), 150)
    Q_grid, H_grid = np.meshgrid(discharge_grid, head_grid)
    
    # Interpolate power values
    P_grid = griddata((discharge, head), power, (Q_grid, H_grid), method='cubic')
    
    # Create 2D contour plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filled contours for power
    levels = np.linspace(np.nanmin(P_grid), np.nanmax(P_grid), 25)
    contourf = ax.contourf(Q_grid, H_grid, P_grid, levels=levels, cmap='plasma', alpha=0.95)
    contour = ax.contour(Q_grid, H_grid, P_grid, levels=levels[::2], colors='white', 
                        linewidths=0.8, alpha=0.6)
    
    # Add contour labels
    ax.clabel(contour, inline=True, fontsize=9, fmt='%.1f kW')
    
    # Plot guide vane angle curves
    unique_gv = sorted(df['G/V degree'].unique())
    gv_colors = plt.cm.cool(np.linspace(0, 1, len(unique_gv)))
    
    for i, gv in enumerate(unique_gv):
        df_gv = df[df['G/V degree'] == gv].sort_values('Head')
        if len(df_gv) > 1:
            ax.plot(df_gv['Dischargem'], df_gv['Head'], 
                   color=gv_colors[i], linewidth=2.5, marker='s', 
                   markersize=6, markeredgecolor='white', markeredgewidth=1.5,
                   label=f'{gv:.0f}°', alpha=0.9, zorder=5)
    
    # Highlight maximum power point
    ax.scatter([max_discharge], [max_head], c='lime', marker='*', 
              s=1200, edgecolors='black', linewidths=3, 
              label='Max Power', zorder=10)
    
    # Add annotation
    ax.annotate(f'Max Power\nP = {max_power:.1f} kW\nQ = {max_discharge:.3f} m³/s\nH = {max_head:.1f} m\nG/V = {max_gv:.1f}°',
               xy=(max_discharge, max_head), xytext=(40, -50),
               textcoords='offset points', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='lime', 
                        edgecolor='black', linewidth=2.5, alpha=0.95),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', 
                             lw=3, color='black'))
    
    # Colorbar
    cbar = plt.colorbar(contourf, ax=ax, pad=0.02)
    cbar.set_label('Generator Power (kW)', fontsize=13, fontweight='bold')
    
    # Labels
    ax.set_xlabel('Discharge Q (m³/s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Head H (m)', fontsize=14, fontweight='bold')
    ax.set_title('Power Output Hill Chart\nBulb Turbine Power Generation Map\n(Lines show constant Guide Vane angles)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    legend = ax.legend(title='Guide Vane\nAngle', loc='best', fontsize=9, 
                      ncol=2, framealpha=0.95, title_fontsize=10)
    legend.get_title().set_fontweight('bold')
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    return fig, ax

def main():
    # Load data
    print("="*70)
    print("TRADITIONAL OPERATING HILL CHART GENERATOR")
    print("="*70)
    print("\nLoading data from bf-bulb folder...")
    df = load_all_data('bf-bulb')
    print(f"Loaded {len(df)} data points from {df['Head'].nunique()} different heads")
    print(f"\nOperating Range:")
    print(f"  Discharge: {df['Dischargem'].min():.3f} - {df['Dischargem'].max():.3f} m³/s")
    print(f"  Head: {df['Head'].min():.1f} - {df['Head'].max():.1f} m")
    print(f"  Guide Vane Angle: {df['G/V degree'].min():.1f}° - {df['G/V degree'].max():.1f}°")
    print(f"  Overall Efficiency: {df['Overall Eff'].min():.4f} - {df['Overall Eff'].max():.4f}")
    
    # Create traditional 3D hill chart
    print("\n" + "="*70)
    print("Creating 3D Traditional Hill Chart (Q vs H vs η)...")
    print("="*70)
    fig1, ax1 = create_traditional_hill_chart_3d(df, metric='Overall Eff')
    fig1.savefig('traditional_hill_chart_3d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: traditional_hill_chart_3d.png")
    
    # Create traditional 2D contour hill chart
    print("\n" + "="*70)
    print("Creating 2D Traditional Hill Chart (Q vs H with efficiency contours)...")
    print("="*70)
    fig2, ax2 = create_traditional_hill_chart_contour(df, metric='Overall Eff')
    fig2.savefig('traditional_hill_chart_2d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: traditional_hill_chart_2d.png")
    
    # Create power output hill chart
    print("\n" + "="*70)
    print("Creating Power Output Hill Chart...")
    print("="*70)
    fig3, ax3 = create_power_hill_chart(df)
    fig3.savefig('power_hill_chart_2d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: power_hill_chart_2d.png")
    
    # Create mechanical efficiency chart
    print("\n" + "="*70)
    print("Creating Mechanical Efficiency Hill Chart...")
    print("="*70)
    fig4, ax4 = create_traditional_hill_chart_contour(df, metric='Mech Eff')
    fig4.savefig('traditional_hill_chart_2d_mech_eff.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: traditional_hill_chart_2d_mech_eff.png")
    
    print("\n" + "="*70)
    print("ALL CHARTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nDisplaying interactive plots...")
    plt.show()

if __name__ == "__main__":
    main()