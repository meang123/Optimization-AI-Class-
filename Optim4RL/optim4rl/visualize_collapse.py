#!/usr/bin/env python3
"""
Visualize PPO Ant Non-Stationary Results and Analyze Policy Collapse
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_ppo_results(result_path):
    """Load PPO baseline results from feather file"""
    df = pd.read_feather(result_path)
    print(f"Loaded {len(df)} data points from {result_path}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def detect_collapse(returns, threshold=0.3):
    """
    Detect policy collapse events
    threshold: percentage drop from moving average to be considered collapse
    """
    # Calculate rolling mean
    window = min(50, len(returns) // 10)
    rolling_mean = pd.Series(returns).rolling(window=window, center=True).mean()

    # Calculate drops from rolling mean
    drops = (returns - rolling_mean) / (rolling_mean + 1e-8)

    # Identify collapse points (significant drops)
    collapse_mask = drops < -threshold

    return collapse_mask, rolling_mean

def analyze_performance_statistics(df):
    """Calculate key performance statistics"""
    returns = df['Return'].values
    steps = df['Step'].values

    stats = {
        'mean': np.mean(returns),
        'std': np.std(returns),
        'median': np.median(returns),
        'max': np.max(returns),
        'min': np.min(returns),
        'cv': np.std(returns) / (np.mean(returns) + 1e-8),  # Coefficient of variation
        'max_step': np.argmax(returns),
        'min_step': np.argmin(returns),
    }

    # Calculate early vs late performance
    split_idx = len(returns) // 2
    early_mean = np.mean(returns[:split_idx])
    late_mean = np.mean(returns[split_idx:])
    stats['early_mean'] = early_mean
    stats['late_mean'] = late_mean
    stats['performance_drop_pct'] = ((early_mean - late_mean) / early_mean) * 100

    return stats

def plot_training_curves(df, save_path='results/ppo_collapse_analysis2.png'):
    """Generate comprehensive training curve visualization"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    returns = df['Return'].values
    steps = df['Step'].values

    # 1. Raw training curve
    ax = axes[0, 0]
    ax.plot(steps, returns, alpha=0.6, linewidth=0.8, label='Episode Returns')

    # Add rolling mean
    window = min(50, len(returns) // 10)
    rolling_mean = pd.Series(returns).rolling(window=window, center=True).mean()
    ax.plot(steps, rolling_mean, 'r-', linewidth=2, label=f'Rolling Mean (w={window})')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title('PPO Ant Non-Stationary: Training Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Collapse detection
    ax = axes[0, 1]
    collapse_mask, rolling_mean = detect_collapse(returns, threshold=0.3)

    # Plot returns with collapse points highlighted
    ax.plot(steps, returns, 'b-', alpha=0.4, linewidth=0.8, label='Returns')
    ax.plot(steps, rolling_mean, 'g-', linewidth=2, label='Trend')

    # Highlight collapse events
    collapse_steps = steps[collapse_mask]
    collapse_returns = returns[collapse_mask]
    ax.scatter(collapse_steps, collapse_returns, c='red', s=50, alpha=0.6,
               label=f'Collapse Events ({np.sum(collapse_mask)})', zorder=5)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title('Policy Collapse Detection (30% drop threshold)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Return distribution
    ax = axes[1, 0]
    ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(returns), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.1f}')
    ax.axvline(np.median(returns), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(returns):.1f}')
    ax.set_xlabel('Episode Return')
    ax.set_ylabel('Frequency')
    ax.set_title('Return Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Performance over training phases
    ax = axes[1, 1]
    n_bins = 10
    bin_size = len(returns) // n_bins
    phase_means = []
    phase_stds = []
    phase_labels = []

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(returns)
        phase_returns = returns[start_idx:end_idx]
        phase_means.append(np.mean(phase_returns))
        phase_stds.append(np.std(phase_returns))
        phase_labels.append(f'Phase {i+1}')

    x_pos = np.arange(n_bins)
    ax.bar(x_pos, phase_means, yerr=phase_stds, alpha=0.7, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phase_labels, rotation=45)
    ax.set_ylabel('Mean Episode Return')
    ax.set_title('Performance Across Training Phases')
    ax.axhline(np.mean(returns), color='r', linestyle='--', alpha=0.5, label='Overall Mean')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Rolling statistics
    ax = axes[2, 0]
    rolling_mean = pd.Series(returns).rolling(window=window).mean()
    rolling_std = pd.Series(returns).rolling(window=window).std()

    ax.plot(steps, rolling_mean, 'b-', linewidth=2, label='Rolling Mean')
    ax.fill_between(steps, rolling_mean - rolling_std, rolling_mean + rolling_std,
                     alpha=0.3, label='±1 Std Dev')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title(f'Rolling Statistics (window={window})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Stability metric (Coefficient of Variation over time)
    ax = axes[2, 1]
    cv_window = max(20, len(returns) // 20)
    rolling_cv = pd.Series(returns).rolling(window=cv_window).std() / (
        pd.Series(returns).rolling(window=cv_window).mean() + 1e-8
    )

    ax.plot(steps, rolling_cv, 'purple', linewidth=2)
    ax.axhline(np.mean(rolling_cv[~np.isnan(rolling_cv)]), color='r',
               linestyle='--', alpha=0.7, label=f'Mean CV: {np.mean(rolling_cv[~np.isnan(rolling_cv)]):.3f}')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title(f'Training Stability (window={cv_window})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to {save_path}")

    return fig

def generate_report(df, save_path='results/ppo_collapse_report2.txt'):
    """Generate detailed text report"""
    stats = analyze_performance_statistics(df)
    
    returns = df['Return'].values
    collapse_mask, rolling_mean = detect_collapse(returns, threshold=0.3)

    report = []
    report.append("=" * 80)
    report.append("PPO ANT NON-STATIONARY: POLICY COLLAPSE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    report.append("📊 OVERALL PERFORMANCE STATISTICS")
    report.append("-" * 80)
    report.append(f"Total Episodes:           {len(returns):,}")
    report.append(f"Total Training Steps:     {df['Step'].max():,.0f}")
    report.append(f"Mean Return:              {stats['mean']:,.2f} ± {stats['std']:,.2f}")
    report.append(f"Median Return:            {stats['median']:,.2f}")
    report.append(f"Max Return:               {stats['max']:,.2f} (at step {df['Step'].iloc[stats['max_step']]:,.0f})")
    report.append(f"Min Return:               {stats['min']:,.2f} (at step {df['Step'].iloc[stats['min_step']]:,.0f})")
    report.append(f"Coefficient of Variation: {stats['cv']:.3f} (σ/μ)")
    report.append("")

    report.append("🔍 POLICY COLLAPSE ANALYSIS")
    report.append("-" * 80)
    collapse_count = np.sum(collapse_mask)
    collapse_rate = (collapse_count / len(returns)) * 100

    report.append(f"Collapse Events Detected: {collapse_count} / {len(returns)} episodes")
    report.append(f"Collapse Rate:            {collapse_rate:.2f}%")
    report.append("")

    # Early vs Late performance
    report.append("📉 PERFORMANCE DEGRADATION")
    report.append("-" * 80)
    report.append(f"Early Training Mean:      {stats['early_mean']:,.2f} (first half)")
    report.append(f"Late Training Mean:       {stats['late_mean']:,.2f} (second half)")
    report.append(f"Performance Drop:         {stats['performance_drop_pct']:.1f}%")

    # Peak to nadir
    peak_to_nadir = ((stats['max'] - stats['min']) / stats['max']) * 100
    report.append(f"Peak to Nadir Drop:       {peak_to_nadir:.1f}%")
    report.append("")

    # Stability assessment
    report.append("📊 STABILITY ASSESSMENT")
    report.append("-" * 80)
    if stats['cv'] < 0.3:
        stability = "HIGH (Stable)"
    elif stats['cv'] < 0.5:
        stability = "MEDIUM (Moderately Stable)"
    else:
        stability = "LOW (Unstable)"

    report.append(f"Stability Rating:         {stability}")
    report.append(f"  - CV < 0.3: High stability")
    report.append(f"  - CV 0.3-0.5: Medium stability")
    report.append(f"  - CV > 0.5: Low stability (CURRENT: {stats['cv']:.3f})")
    report.append("")

    # Collapse severity
    report.append("⚠️  COLLAPSE SEVERITY")
    report.append("-" * 80)
    if stats['performance_drop_pct'] > 50:
        severity = "SEVERE (>50% drop)"
    elif stats['performance_drop_pct'] > 30:
        severity = "MODERATE (30-50% drop)"
    elif stats['performance_drop_pct'] > 15:
        severity = "MILD (15-30% drop)"
    else:
        severity = "MINIMAL (<15% drop)"

    report.append(f"Collapse Severity:        {severity}")
    report.append(f"  - >50%: Severe collapse")
    report.append(f"  - 30-50%: Moderate collapse")
    report.append(f"  - 15-30%: Mild degradation")
    report.append(f"  - <15%: Minimal impact")
    report.append(f"  - CURRENT: {stats['performance_drop_pct']:.1f}% drop")
    report.append("")

    report.append("=" * 80)
    report.append("🎯 CONCLUSION")
    report.append("=" * 80)

    if stats['performance_drop_pct'] > 30 and stats['cv'] > 0.5:
        conclusion = "❌ POLICY COLLAPSE CONFIRMED: Severe performance degradation and high instability"
    elif stats['performance_drop_pct'] > 15 or stats['cv'] > 0.4:
        conclusion = "⚠️  POLICY DEGRADATION OBSERVED: Moderate performance issues detected"
    else:
        conclusion = "✅ STABLE TRAINING: No significant policy collapse detected"

    report.append(conclusion)
    report.append("")
    report.append("Key Findings:")
    report.append(f"  • Performance dropped {stats['performance_drop_pct']:.1f}% from early to late training")
    report.append(f"  • {collapse_count} collapse events detected ({collapse_rate:.1f}% of episodes)")
    report.append(f"  • Training stability is {stability.split('(')[1].rstrip(')')}")
    report.append(f"  • Peak performance: {stats['max']:,.0f}, Worst: {stats['min']:,.0f}")
    report.append("")

    report_text = "\n".join(report)

    # Save report
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n✅ Saved report to {save_path}")

    return report_text

def main():
    # Load PPO baseline results
    ppo_result_path = "logs/ppo_ant_nonstationary_test_learned/1/result_Test.feather"

    print("=" * 80)
    print("LOADING PPO ANT NON-STATIONARY RESULTS")
    print("=" * 80)

    df = load_ppo_results(ppo_result_path)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    plot_training_curves(df)

    # Generate report
    print("\n" + "=" * 80)
    print("GENERATING ANALYSIS REPORT")
    print("=" * 80)
    generate_report(df)

    # Note about MetaPPO
    print("\n" + "=" * 80)
    print("⚠️  NOTE: MetaPPO PERFORMANCE DATA")
    print("=" * 80)
    print("MetaPPO meta-training completed successfully, but agent performance")
    print("metrics are not available. Meta-training logs only track optimizer")
    print("parameter updates, not episode returns.")
    print("")
    print("To compare MetaPPO with PPO baseline, you would need to:")
    print("1. Create a test configuration using learned optimizer")
    print("2. Run evaluation with: python main.py --config_file <test_config> --config_idx 1")
    print("3. Re-run this analysis script to compare both results")
    print("=" * 80)

if __name__ == "__main__":
    main()
