"""
Golden Ratio-Based Sampling for Hill Chart Analysis
Nature-Inspired Data-Efficient Turbine Performance Reconstruction
Author: Your Name
Date: 2025
"""

import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from src.data_loader import load_all_data
from src.fibonacci import simulate_fibonacci_spiral_sampling
from src.bayesian import simulate_bayesian_sampling
from src.random_sampling import simulate_random_sampling
from src.grid import simulate_grid_sampling
from src.lhs import simulate_lhs_sampling
from src.metrics import calculate_metrics, find_bep
from src.visualizations import (
    create_ground_truth_3d,
    create_fibonacci_3d,
    create_error_3d,
    create_ground_truth_2d,
    create_fibonacci_2d,
    create_error_2d,
    create_method_comparison_surfaces,
    create_bep_comparison_all_methods,
    create_bep_comparison_table_all_methods,
    create_method_comparison_metrics,
    create_method_comparison_table
)

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


def main():
    """Main execution function"""
    
    # Create result directory
    result_dir = Path('result')
    result_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {result_dir.absolute()}")
    
    # Load data
    print("\nLoading ground truth data from bf-gv folder...")
    df = load_all_data('bf-gv')
    print(f"Loaded {len(df)} experiments")
    
    # Parameters
    n_samples = 30  # Target number of samples for all methods
    
    print(f"\n{'='*70}")
    print(f"GOLDEN RATIO-BASED SAMPLING ANALYSIS")
    print(f"Nature-Inspired Data-Efficient Approach")
    print(f"{'='*70}")
    print(f"Comparing {n_samples} samples each method")
    
    # Run all methods - FIBONACCI FIRST (primary method)
    print("\n1. Golden Ratio (Fibonacci Spiral) Sampling...")
    fib_indices, fib_pred, fib_std = simulate_fibonacci_spiral_sampling(
        df, n_samples, 'Overall Eff')
    fib_metrics = calculate_metrics(df, fib_pred, 'Overall Eff')
    
    print("\n2. Bayesian Optimization...")
    n_initial = 5
    n_iterations = n_samples - n_initial
    bayesian_indices, bayesian_pred, bayesian_std = simulate_bayesian_sampling(
        df, n_initial, n_iterations, 'Overall Eff')
    bayesian_metrics = calculate_metrics(df, bayesian_pred, 'Overall Eff')
    
    print("\n3. Random Sampling...")
    random_indices, random_pred, random_std = simulate_random_sampling(
        df, n_samples, 'Overall Eff')
    random_metrics = calculate_metrics(df, random_pred, 'Overall Eff')
    
    print("\n4. Grid Sampling...")
    grid_indices, grid_pred, grid_std = simulate_grid_sampling(
        df, n_samples, 'Overall Eff')
    grid_metrics = calculate_metrics(df, grid_pred, 'Overall Eff')
    
    print("\n5. Latin Hypercube Sampling...")
    lhs_indices, lhs_pred, lhs_std = simulate_lhs_sampling(
        df, n_samples, 'Overall Eff')
    lhs_metrics = calculate_metrics(df, lhs_pred, 'Overall Eff')
    
    # Compile results - FIBONACCI FIRST
    method_results = {
        'Fibonacci': {'indices': fib_indices, 'predictions': fib_pred, **fib_metrics},
        'Bayesian': {'indices': bayesian_indices, 'predictions': bayesian_pred, **bayesian_metrics},
        'Random': {'indices': random_indices, 'predictions': random_pred, **random_metrics},
        'Grid': {'indices': grid_indices, 'predictions': grid_pred, **grid_metrics},
        'LHS': {'indices': lhs_indices, 'predictions': lhs_pred, **lhs_metrics}
    }
    
    # Find BEP for each method
    for method in ['Fibonacci', 'Bayesian', 'Random', 'Grid', 'LHS']:
        bep = find_bep(df, method_results[method]['indices'], 
                      method_results[method]['predictions'])
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
    for method in ['Fibonacci', 'Bayesian', 'Random', 'Grid', 'LHS']:
        bep = method_results[method]['bep']
        method_results[method]['bep_error'] = {
            'discharge': abs(bep['discharge'] - bep_ground_truth[0]),
            'head': abs(bep['head'] - bep_ground_truth[1]),
            'efficiency': abs(bep['efficiency'] - bep_ground_truth[2])
        }
    
    # Generate all figures
    print(f"\n{'='*70}")
    print("Generating figures...")
    print(f"{'='*70}")
    
    # Figures 1-6: Golden Ratio Method Details
    fig1, _ = create_ground_truth_3d(df, 'Overall Eff')
    fig1.savefig(result_dir / 'fig1_ground_truth_3d.png', dpi=600, bbox_inches='tight')
    fig1.savefig(result_dir / 'fig1_ground_truth_3d.pdf', bbox_inches='tight')
    print("✓ fig1_ground_truth_3d")
    
    fig2 = create_fibonacci_3d(df, fib_indices, fib_pred, bep_ground_truth, 'Overall Eff')
    fig2.savefig(result_dir / 'fig2_fibonacci_3d.png', dpi=600, bbox_inches='tight')
    fig2.savefig(result_dir / 'fig2_fibonacci_3d.pdf', bbox_inches='tight')
    print("✓ fig2_fibonacci_3d (Golden Ratio)")
    
    fig3 = create_error_3d(df, fib_indices, fib_pred, 'Overall Eff')
    fig3.savefig(result_dir / 'fig3_error_3d.png', dpi=600, bbox_inches='tight')
    fig3.savefig(result_dir / 'fig3_error_3d.pdf', bbox_inches='tight')
    print("✓ fig3_error_3d")
    
    fig4 = create_ground_truth_2d(df, 'Overall Eff')
    fig4.savefig(result_dir / 'fig4_ground_truth_2d.png', dpi=600, bbox_inches='tight')
    fig4.savefig(result_dir / 'fig4_ground_truth_2d.pdf', bbox_inches='tight')
    print("✓ fig4_ground_truth_2d")
    
    fig5 = create_fibonacci_2d(df, fib_indices, fib_pred, bep_ground_truth, 'Overall Eff')
    fig5.savefig(result_dir / 'fig5_fibonacci_2d.png', dpi=600, bbox_inches='tight')
    fig5.savefig(result_dir / 'fig5_fibonacci_2d.pdf', bbox_inches='tight')
    print("✓ fig5_fibonacci_2d (Golden Ratio)")
    
    fig6 = create_error_2d(df, fib_indices, fib_pred, 'Overall Eff')
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
    
    # Save data files
    print(f"\nSaving data files...")
    for method_name, method_data in method_results.items():
        method_df = df.iloc[method_data['indices']].copy()
        method_df['Predicted_Eff'] = method_data['predictions'][method_data['indices']]
        method_df.to_csv(result_dir / f'{method_name.lower()}_sampled_points.csv', index=False)
    print("✓ Sampled points for all methods")
    
    # Save summary
    summary = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'study_focus': 'Golden Ratio (Fibonacci Spiral) Sampling',
            'total_experiments': len(df),
            'samples_per_method': n_samples,
            'reduction_percentage': (1 - n_samples/len(df)) * 100,
            'golden_ratio': 1.618033988749895,
            'golden_angle_degrees': 137.507764
        },
        'bep_ground_truth': {
            'discharge': float(bep_ground_truth[0]),
            'head': float(bep_ground_truth[1]),
            'efficiency': float(bep_ground_truth[2])
        },
        'methods': {}
    }
    
    for method in ['Fibonacci', 'Bayesian', 'Random', 'Grid', 'LHS']:
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
    
    # Print results summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY - GOLDEN RATIO FOCUS")
    print(f"{'='*70}")
    print(f"\nExperiment Reduction: {len(df)} → {n_samples} ({(1-n_samples/len(df))*100:.1f}%)")
    print(f"\nMethod Comparison (Golden Ratio is primary method):")
    print(f"{'Method':<12} {'MAE':<10} {'R²':<10} {'BEP Error':<12} {'Rank'}")
    print("-" * 54)
    
    # Rank methods by MAE
    mae_ranks = sorted(method_results.items(), key=lambda x: x[1]['mae'])
    
    for rank, (method, data) in enumerate(mae_ranks, 1):
        bep_total = sum(data['bep_error'].values())
        marker = "⭐" if method == "Fibonacci" else ""
        print(f"{method:<12} {data['mae']:<10.5f} {data['r2']:<10.4f} {bep_total:<12.5f} #{rank} {marker}")
    
    fib_rank = next(i for i, (m, _) in enumerate(mae_ranks, 1) if m == "Fibonacci")
    
    if fib_rank == 1:
        print(f"\n🎉 Golden Ratio is BEST! (Rank #{fib_rank})")
    else:
        best_method = mae_ranks[0][0]
        improvement_needed = ((method_results[best_method]['mae'] - fib_metrics['mae']) / 
                             fib_metrics['mae']) * 100
        print(f"\n✨ Golden Ratio ranks #{fib_rank}")
        print(f"   {improvement_needed:.1f}% difference from best method ({best_method})")
    
    print(f"\n{'='*70}")
    print(f"All results saved to: {result_dir.absolute()}")
    print(f"{'='*70}")
    
    plt.show()


if __name__ == "__main__":
    main()
