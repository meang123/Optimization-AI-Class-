#!/usr/bin/env python3
"""
Analyze PPO Ant Non-Stationary Adam Results
Compare with standard Adam optimizer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

def load_results(result_path):
    """Load PPO results from feather file"""
    df = pd.read_feather(result_path)
    print(f"Loaded {len(df)} data points from {result_path}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def analyze_performance_statistics(df, label=""):
    """Calculate key performance statistics"""
    returns = df['Return'].values
    steps = df['Step'].values

    stats = {
        'label': label,
        'mean': np.mean(returns),
        'std': np.std(returns),
        'median': np.median(returns),
        'max': np.max(returns),
        'min': np.min(returns),
        'cv': np.std(returns) / (np.mean(returns) + 1e-8),
        'max_step': np.argmax(returns),
        'min_step': np.argmin(returns),
    }

    # Calculate early vs late performance
    split_idx = len(returns) // 2
    early_mean = np.mean(returns[:split_idx])
    late_mean = np.mean(returns[split_idx:])
    stats['early_mean'] = early_mean
    stats['late_mean'] = late_mean
    stats['performance_drop_pct'] = ((early_mean - late_mean) / (early_mean + 1e-8)) * 100

    return stats

def detect_collapse(returns, threshold=0.3):
    """Detect policy collapse events"""
    window = min(50, len(returns) // 10)
    rolling_mean = pd.Series(returns).rolling(window=window, center=True).mean()
    drops = (returns - rolling_mean) / (rolling_mean + 1e-8)
    collapse_mask = drops < -threshold
    return collapse_mask, rolling_mean

def plot_comparison(df_standard, df_adam, save_path='results/ppo_ant_adam_comparison.png'):
    """Generate comparison visualization"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    returns_std = df_standard['Return'].values
    steps_std = df_standard['Step'].values
    returns_adam = df_adam['Return'].values
    steps_adam = df_adam['Step'].values

    # 1. Training curves comparison
    ax = axes[0, 0]
    ax.plot(steps_std, returns_std, alpha=0.5, linewidth=0.8, label='Standard (default Adam)', color='blue')
    ax.plot(steps_adam, returns_adam, alpha=0.5, linewidth=0.8, label='Adam (b1=b2=0.997)', color='red')

    # Add rolling means
    window = min(5, len(returns_std) // 10)
    rm_std = pd.Series(returns_std).rolling(window=window, center=True).mean()
    rm_adam = pd.Series(returns_adam).rolling(window=window, center=True).mean()
    ax.plot(steps_std, rm_std, 'b-', linewidth=2, label=f'Standard Rolling Mean (w={window})')
    ax.plot(steps_adam, rm_adam, 'r-', linewidth=2, label=f'Adam Rolling Mean (w={window})')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title('Training Curves Comparison: Standard vs Adam (b1=b2=0.997)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Distribution comparison
    ax = axes[0, 1]
    ax.hist(returns_std, bins=30, alpha=0.6, label='Standard', color='blue', edgecolor='black')
    ax.hist(returns_adam, bins=30, alpha=0.6, label='Adam (b1=b2=0.997)', color='red', edgecolor='black')
    ax.axvline(np.mean(returns_std), color='blue', linestyle='--', linewidth=2, label=f'Std Mean: {np.mean(returns_std):.1f}')
    ax.axvline(np.mean(returns_adam), color='red', linestyle='--', linewidth=2, label=f'Adam Mean: {np.mean(returns_adam):.1f}')
    ax.set_xlabel('Episode Return')
    ax.set_ylabel('Frequency')
    ax.set_title('Return Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Collapse detection comparison
    ax = axes[1, 0]
    collapse_std, rm_std_full = detect_collapse(returns_std, threshold=0.3)
    collapse_adam, rm_adam_full = detect_collapse(returns_adam, threshold=0.3)

    ax.plot(steps_std, returns_std, 'b-', alpha=0.3, linewidth=0.8, label='Standard')
    ax.plot(steps_adam, returns_adam, 'r-', alpha=0.3, linewidth=0.8, label='Adam (b1=b2=0.997)')
    ax.plot(steps_std, rm_std_full, 'b-', linewidth=2, label='Standard Trend')
    ax.plot(steps_adam, rm_adam_full, 'r-', linewidth=2, label='Adam Trend')

    # Highlight collapse events
    if np.sum(collapse_std) > 0:
        ax.scatter(steps_std[collapse_std], returns_std[collapse_std],
                   c='blue', s=100, alpha=0.7, marker='x', label=f'Std Collapse ({np.sum(collapse_std)})', zorder=5)
    if np.sum(collapse_adam) > 0:
        ax.scatter(steps_adam[collapse_adam], returns_adam[collapse_adam],
                   c='red', s=100, alpha=0.7, marker='x', label=f'Adam Collapse ({np.sum(collapse_adam)})', zorder=5)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title('Policy Collapse Detection (30% drop threshold)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Performance by phase
    ax = axes[1, 1]
    n_bins = 10
    bin_size_std = len(returns_std) // n_bins
    bin_size_adam = len(returns_adam) // n_bins

    phase_means_std = []
    phase_means_adam = []

    for i in range(n_bins):
        start_idx = i * bin_size_std
        end_idx = (i + 1) * bin_size_std if i < n_bins - 1 else len(returns_std)
        phase_means_std.append(np.mean(returns_std[start_idx:end_idx]))

        start_idx = i * bin_size_adam
        end_idx = (i + 1) * bin_size_adam if i < n_bins - 1 else len(returns_adam)
        phase_means_adam.append(np.mean(returns_adam[start_idx:end_idx]))

    x_pos = np.arange(n_bins)
    width = 0.35
    ax.bar(x_pos - width/2, phase_means_std, width, alpha=0.7, label='Standard', color='blue')
    ax.bar(x_pos + width/2, phase_means_adam, width, alpha=0.7, label='Adam (b1=b2=0.997)', color='red')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'P{i+1}' for i in range(n_bins)])
    ax.set_ylabel('Mean Episode Return')
    ax.set_title('Performance Across Training Phases')
    ax.axhline(np.mean(returns_std), color='blue', linestyle='--', alpha=0.5, label='Std Overall Mean')
    ax.axhline(np.mean(returns_adam), color='red', linestyle='--', alpha=0.5, label='Adam Overall Mean')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Stability (CV over time)
    ax = axes[2, 0]
    cv_window = max(5, len(returns_std) // 20)

    rolling_cv_std = pd.Series(returns_std).rolling(window=cv_window).std() / (
        pd.Series(returns_std).rolling(window=cv_window).mean() + 1e-8
    )
    rolling_cv_adam = pd.Series(returns_adam).rolling(window=cv_window).std() / (
        pd.Series(returns_adam).rolling(window=cv_window).mean() + 1e-8
    )

    ax.plot(steps_std, rolling_cv_std, 'blue', linewidth=2, label='Standard', alpha=0.7)
    ax.plot(steps_adam, rolling_cv_adam, 'red', linewidth=2, label='Adam (b1=b2=0.997)', alpha=0.7)

    mean_cv_std = np.mean(rolling_cv_std[~np.isnan(rolling_cv_std)])
    mean_cv_adam = np.mean(rolling_cv_adam[~np.isnan(rolling_cv_adam)])

    ax.axhline(mean_cv_std, color='blue', linestyle='--', alpha=0.5, label=f'Std Mean CV: {mean_cv_std:.3f}')
    ax.axhline(mean_cv_adam, color='red', linestyle='--', alpha=0.5, label=f'Adam Mean CV: {mean_cv_adam:.3f}')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title(f'Training Stability (window={cv_window})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Statistics summary table
    ax = axes[2, 1]
    ax.axis('off')

    stats_std = analyze_performance_statistics(df_standard, "Standard")
    stats_adam = analyze_performance_statistics(df_adam, "Adam (b1=b2=0.997)")

    table_data = [
        ['Metric', 'Standard', 'Adam (b1=b2=0.997)', 'Difference'],
        ['Mean Return', f"{stats_std['mean']:.1f}", f"{stats_adam['mean']:.1f}",
         f"{stats_adam['mean'] - stats_std['mean']:+.1f}"],
        ['Std Dev', f"{stats_std['std']:.1f}", f"{stats_adam['std']:.1f}",
         f"{stats_adam['std'] - stats_std['std']:+.1f}"],
        ['Max Return', f"{stats_std['max']:.1f}", f"{stats_adam['max']:.1f}",
         f"{stats_adam['max'] - stats_std['max']:+.1f}"],
        ['Min Return', f"{stats_std['min']:.1f}", f"{stats_adam['min']:.1f}",
         f"{stats_adam['min'] - stats_std['min']:+.1f}"],
        ['CV (Stability)', f"{stats_std['cv']:.3f}", f"{stats_adam['cv']:.3f}",
         f"{stats_adam['cv'] - stats_std['cv']:+.3f}"],
        ['Early Mean', f"{stats_std['early_mean']:.1f}", f"{stats_adam['early_mean']:.1f}",
         f"{stats_adam['early_mean'] - stats_std['early_mean']:+.1f}"],
        ['Late Mean', f"{stats_std['late_mean']:.1f}", f"{stats_adam['late_mean']:.1f}",
         f"{stats_adam['late_mean'] - stats_std['late_mean']:+.1f}"],
        ['Performance Drop', f"{stats_std['performance_drop_pct']:.1f}%",
         f"{stats_adam['performance_drop_pct']:.1f}%",
         f"{stats_adam['performance_drop_pct'] - stats_std['performance_drop_pct']:+.1f}%"],
        ['Collapse Events', f"{np.sum(collapse_std)}", f"{np.sum(collapse_adam)}",
         f"{np.sum(collapse_adam) - np.sum(collapse_std):+d}"],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.23, 0.23, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Performance Statistics Comparison', pad=20, fontsize=12, weight='bold')

    plt.tight_layout()

    # Save figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comparison visualization to {save_path}")

    return fig

def generate_comparison_report(df_standard, df_adam, save_path='results/ppo_ant_adam_comparison_report.txt'):
    """Generate detailed comparison report"""
    stats_std = analyze_performance_statistics(df_standard, "Standard")
    stats_adam = analyze_performance_statistics(df_adam, "Adam (b1=b2=0.997)")

    returns_std = df_standard['Return'].values
    returns_adam = df_adam['Return'].values
    collapse_std, _ = detect_collapse(returns_std, threshold=0.3)
    collapse_adam, _ = detect_collapse(returns_adam, threshold=0.3)

    report = []
    report.append("=" * 80)
    report.append("PPO ANT NON-STATIONARY: ADAM OPTIMIZER COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")

    report.append("📊 OPTIMIZER CONFIGURATIONS")
    report.append("-" * 80)
    report.append("Standard Adam:")
    report.append("  - Learning Rate: 3e-4")
    report.append("  - b1: 0.9 (default)")
    report.append("  - b2: 0.999 (default)")
    report.append("")
    report.append("Modified Adam:")
    report.append("  - Learning Rate: 3e-4")
    report.append("  - b1: 0.997")
    report.append("  - b2: 0.997")
    report.append("")

    report.append("📈 OVERALL PERFORMANCE STATISTICS")
    report.append("-" * 80)
    report.append(f"{'Metric':<25} {'Standard':<20} {'Adam (b1=b2=0.997)':<20} {'Difference':<15}")
    report.append("-" * 80)
    report.append(f"{'Total Episodes':<25} {len(returns_std):<20} {len(returns_adam):<20} {len(returns_adam) - len(returns_std):<15}")
    report.append(f"{'Mean Return':<25} {stats_std['mean']:,.2f}{'':<12} {stats_adam['mean']:,.2f}{'':<12} {stats_adam['mean'] - stats_std['mean']:+,.2f}")
    report.append(f"{'Std Dev':<25} {stats_std['std']:,.2f}{'':<12} {stats_adam['std']:,.2f}{'':<12} {stats_adam['std'] - stats_std['std']:+,.2f}")
    report.append(f"{'Max Return':<25} {stats_std['max']:,.2f}{'':<12} {stats_adam['max']:,.2f}{'':<12} {stats_adam['max'] - stats_std['max']:+,.2f}")
    report.append(f"{'Min Return':<25} {stats_std['min']:,.2f}{'':<12} {stats_adam['min']:,.2f}{'':<12} {stats_adam['min'] - stats_std['min']:+,.2f}")
    report.append(f"{'CV (Stability)':<25} {stats_std['cv']:.3f}{'':<16} {stats_adam['cv']:.3f}{'':<16} {stats_adam['cv'] - stats_std['cv']:+.3f}")
    report.append("")

    report.append("📉 PERFORMANCE DEGRADATION")
    report.append("-" * 80)
    report.append(f"{'Metric':<25} {'Standard':<20} {'Adam (b1=b2=0.997)':<20} {'Difference':<15}")
    report.append("-" * 80)
    report.append(f"{'Early Training Mean':<25} {stats_std['early_mean']:,.2f}{'':<12} {stats_adam['early_mean']:,.2f}{'':<12} {stats_adam['early_mean'] - stats_std['early_mean']:+,.2f}")
    report.append(f"{'Late Training Mean':<25} {stats_std['late_mean']:,.2f}{'':<12} {stats_adam['late_mean']:,.2f}{'':<12} {stats_adam['late_mean'] - stats_std['late_mean']:+,.2f}")
    report.append(f"{'Performance Drop':<25} {stats_std['performance_drop_pct']:.1f}%{'':<15} {stats_adam['performance_drop_pct']:.1f}%{'':<15} {stats_adam['performance_drop_pct'] - stats_std['performance_drop_pct']:+.1f}%")
    report.append("")

    report.append("⚠️  POLICY COLLAPSE ANALYSIS")
    report.append("-" * 80)
    collapse_count_std = np.sum(collapse_std)
    collapse_count_adam = np.sum(collapse_adam)
    collapse_rate_std = (collapse_count_std / len(returns_std)) * 100
    collapse_rate_adam = (collapse_count_adam / len(returns_adam)) * 100

    report.append(f"{'Metric':<25} {'Standard':<20} {'Adam (b1=b2=0.997)':<20} {'Difference':<15}")
    report.append("-" * 80)
    report.append(f"{'Collapse Events':<25} {collapse_count_std:<20} {collapse_count_adam:<20} {collapse_count_adam - collapse_count_std:+d}")
    report.append(f"{'Collapse Rate':<25} {collapse_rate_std:.2f}%{'':<15} {collapse_rate_adam:.2f}%{'':<15} {collapse_rate_adam - collapse_rate_std:+.2f}%")
    report.append("")

    report.append("=" * 80)
    report.append("🎯 CONCLUSION")
    report.append("=" * 80)

    # Determine winner
    if stats_adam['mean'] > stats_std['mean'] * 1.05:
        conclusion = "✅ ADAM (b1=b2=0.997) SUPERIOR: Significantly better mean performance"
    elif stats_std['mean'] > stats_adam['mean'] * 1.05:
        conclusion = "✅ STANDARD ADAM SUPERIOR: Significantly better mean performance"
    else:
        conclusion = "➖ SIMILAR PERFORMANCE: No significant difference in mean return"

    report.append(conclusion)
    report.append("")

    report.append("Key Findings:")
    report.append(f"  • Mean return difference: {stats_adam['mean'] - stats_std['mean']:+.2f} ({((stats_adam['mean'] - stats_std['mean']) / stats_std['mean'] * 100):+.1f}%)")
    report.append(f"  • Stability (CV): Standard {stats_std['cv']:.3f} vs Adam {stats_adam['cv']:.3f}")
    report.append(f"  • Collapse events: Standard {collapse_count_std} vs Adam {collapse_count_adam}")
    report.append(f"  • Performance drop: Standard {stats_std['performance_drop_pct']:.1f}% vs Adam {stats_adam['performance_drop_pct']:.1f}%")
    report.append("")

    report_text = "\n".join(report)

    # Save report
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n✅ Saved comparison report to {save_path}")

    return report_text

def main():
    # Load both results
    print("=" * 80)
    print("LOADING RESULTS")
    print("=" * 80)

    standard_path = "logs/ppo_ant_nonstationary/1/result_Test.feather"
    adam_path = "logs/ppo_ant_nonstationary_adam/1/result_Test.feather"

    df_standard = load_results(standard_path)
    df_adam = load_results(adam_path)

    # Generate comparison visualization
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON VISUALIZATION")
    print("=" * 80)
    plot_comparison(df_standard, df_adam)

    # Generate comparison report
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON REPORT")
    print("=" * 80)
    generate_comparison_report(df_standard, df_adam)

if __name__ == "__main__":
    main()
