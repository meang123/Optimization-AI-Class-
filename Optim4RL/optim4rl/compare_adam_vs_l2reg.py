#!/usr/bin/env python3
"""
Compare Adam baseline vs L2-regularized Learned Optimizer
Analysis of policy collapse and performance comparison
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path

def load_feather_results(path):
    """Load feather format results"""
    try:
        df = pd.read_feather(path)
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def detect_policy_collapse_advanced(returns, steps=None, threshold_percentile=10, window_size=5):
    """
    Advanced policy collapse detection with multiple indicators
    """
    if len(returns) < window_size:
        return False, "Not enough data", {}

    # Calculate statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)
    median_return = np.median(returns)

    # Detailed metrics
    metrics = {
        'mean': mean_return,
        'std': std_return,
        'min': min_return,
        'max': max_return,
        'median': median_return,
        'cv': std_return / abs(mean_return) if mean_return != 0 else np.inf,  # Coefficient of variation
        'range': max_return - min_return
    }

    # Collapse indicators
    collapse_indicators = []
    severity_score = 0

    # 1. Very low mean return
    if mean_return < 0:
        collapse_indicators.append(f"Negative mean return: {mean_return:.2f}")
        severity_score += 3
    elif mean_return < 500:
        collapse_indicators.append(f"Low mean return: {mean_return:.2f}")
        severity_score += 1

    # 2. Very low median
    if median_return < -200:
        collapse_indicators.append(f"Very low median: {median_return:.2f}")
        severity_score += 3
    elif median_return < 500:
        collapse_indicators.append(f"Low median: {median_return:.2f}")
        severity_score += 1

    # 3. High variance (coefficient of variation > 0.5)
    if metrics['cv'] > 0.8:
        collapse_indicators.append(f"Very high variance (CV={metrics['cv']:.2f})")
        severity_score += 3
    elif metrics['cv'] > 0.5:
        collapse_indicators.append(f"High variance (CV={metrics['cv']:.2f})")
        severity_score += 2

    # 4. Sustained low performance (>50% of returns below threshold)
    threshold = np.percentile(returns, 20)
    low_return_ratio = np.sum(returns < threshold) / len(returns)
    if low_return_ratio > 0.5:
        collapse_indicators.append(f"Sustained low performance: {low_return_ratio:.1%} below threshold")
        severity_score += 2

    # 5. Trend analysis - check if performance degrades over time
    if steps is not None and len(steps) == len(returns):
        # Split into first and second half
        mid_idx = len(returns) // 2
        first_half_mean = np.mean(returns[:mid_idx])
        second_half_mean = np.mean(returns[mid_idx:])

        if first_half_mean - second_half_mean > 500:
            collapse_indicators.append(f"Severe degradation: {first_half_mean:.0f} → {second_half_mean:.0f}")
            severity_score += 4
        elif first_half_mean - second_half_mean > 200:
            collapse_indicators.append(f"Performance degradation: {first_half_mean:.0f} → {second_half_mean:.0f}")
            severity_score += 2

    # 6. Check for sudden drops
    if len(returns) > window_size:
        for i in range(window_size, min(len(returns), window_size + 10)):
            prev_window = returns[i-window_size:i]
            if returns[i] < np.mean(prev_window) - 1000:
                collapse_indicators.append(f"Sudden drop detected at evaluation {i}")
                severity_score += 2
                break

    metrics['severity_score'] = severity_score
    has_collapse = severity_score >= 3

    return has_collapse, collapse_indicators, metrics

def analyze_experiment_detailed(exp_name, log_path):
    """Detailed analysis of a single experiment"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {exp_name}")
    print(f"{'='*80}")

    # Load test results
    result_path = os.path.join(log_path, "result_Test.feather")
    if not os.path.exists(result_path):
        print(f"❌ No test results found at {result_path}")
        return None

    df = load_feather_results(result_path)
    if df is None:
        return None

    print(f"\n📊 Data Overview:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    # Extract returns and steps
    returns = df['Return'].values
    steps = df['Step'].values

    print(f"\n📈 Basic Statistics:")
    print(f"  Total evaluations: {len(returns)}")
    print(f"  Training steps range: {steps[0]:,} → {steps[-1]:,}")
    print(f"  Mean return: {np.mean(returns):.2f}")
    print(f"  Std return: {np.std(returns):.2f}")
    print(f"  Min return: {np.min(returns):.2f}")
    print(f"  Max return: {np.max(returns):.2f}")
    print(f"  Median return: {np.median(returns):.2f}")

    # Percentiles
    print(f"\n📊 Percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p:2d}th: {np.percentile(returns, p):8.2f}")

    # Detect collapse
    has_collapse, indicators, metrics = detect_policy_collapse_advanced(returns, steps)

    print(f"\n🔍 Policy Collapse Analysis:")
    print(f"  Severity Score: {metrics['severity_score']}/15")
    print(f"  Coefficient of Variation: {metrics['cv']:.3f}")

    if has_collapse:
        print(f"  Status: ⚠️  POLICY COLLAPSE DETECTED!")
        print(f"\n  Collapse Indicators ({len(indicators)}):")
        for i, indicator in enumerate(indicators, 1):
            print(f"    {i}. {indicator}")
    else:
        print(f"  Status: ✅ No significant policy collapse")
        if indicators:
            print(f"\n  Minor concerns ({len(indicators)}):")
            for i, indicator in enumerate(indicators, 1):
                print(f"    {i}. {indicator}")

    # Performance over time analysis
    print(f"\n📉 Performance Trajectory:")
    split_points = [0, len(returns)//4, len(returns)//2, 3*len(returns)//4, len(returns)]
    labels = ['Early (0-25%)', 'Mid-Early (25-50%)', 'Mid-Late (50-75%)', 'Late (75-100%)']

    for i in range(len(split_points)-1):
        segment = returns[split_points[i]:split_points[i+1]]
        if len(segment) > 0:
            print(f"  {labels[i]:20s}: {np.mean(segment):8.2f} ± {np.std(segment):7.2f}")

    # Check inference pickle
    inference_pickle_path = os.path.join(log_path, "inference_param_iter5.pickle")
    has_inference = os.path.exists(inference_pickle_path)

    print(f"\n📦 Artifacts:")
    print(f"  Inference pickle (iter5): {'✓' if has_inference else '✗'}")

    return {
        'name': exp_name,
        'returns': returns,
        'steps': steps,
        'mean': np.mean(returns),
        'std': np.std(returns),
        'min': np.min(returns),
        'max': np.max(returns),
        'median': np.median(returns),
        'has_collapse': has_collapse,
        'collapse_indicators': indicators,
        'metrics': metrics,
        'df': df
    }

def create_detailed_comparison_plot(exp1_data, exp2_data, save_path='adam_vs_l2reg_comparison.png'):
    """Create comprehensive comparison visualization"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Returns over training steps (main plot)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(exp1_data['steps'], exp1_data['returns'],
             label=exp1_data['name'], marker='o', markersize=5, alpha=0.7, linewidth=2, color='#FF6B6B')
    ax1.plot(exp2_data['steps'], exp2_data['returns'],
             label=exp2_data['name'], marker='s', markersize=5, alpha=0.7, linewidth=2, color='#4ECDC4')

    # Add horizontal lines for means
    ax1.axhline(y=exp1_data['mean'], color='#FF6B6B', linestyle='--', alpha=0.5, linewidth=1.5, label=f'{exp1_data["name"]} mean')
    ax1.axhline(y=exp2_data['mean'], color='#4ECDC4', linestyle='--', alpha=0.5, linewidth=1.5, label=f'{exp2_data["name"]} mean')

    ax1.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Episode Return', fontsize=13, fontweight='bold')
    ax1.set_title('Episode Returns Over Training (Non-Stationary Ant Environment)', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

    # Plot 2: Box plot comparison
    ax2 = fig.add_subplot(gs[0, 2])
    box_data = [exp1_data['returns'], exp2_data['returns']]
    bp = ax2.boxplot(box_data, tick_labels=[exp1_data['name'][:15], exp2_data['name'][:15]],
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('#FF6B6B')
    bp['boxes'][1].set_facecolor('#4ECDC4')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax2.set_ylabel('Episode Return', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=15)

    # Plot 3: Histogram comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(exp1_data['returns'], bins=20, alpha=0.6, label=exp1_data['name'], color='#FF6B6B', edgecolor='black')
    ax3.hist(exp2_data['returns'], bins=20, alpha=0.6, label=exp2_data['name'], color='#4ECDC4', edgecolor='black')
    ax3.set_xlabel('Episode Return', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Return Distribution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Cumulative distribution
    ax4 = fig.add_subplot(gs[1, 1])
    sorted_returns1 = np.sort(exp1_data['returns'])
    sorted_returns2 = np.sort(exp2_data['returns'])
    cumulative1 = np.arange(1, len(sorted_returns1) + 1) / len(sorted_returns1)
    cumulative2 = np.arange(1, len(sorted_returns2) + 1) / len(sorted_returns2)

    ax4.plot(sorted_returns1, cumulative1, label=exp1_data['name'], linewidth=2.5, color='#FF6B6B')
    ax4.plot(sorted_returns2, cumulative2, label=exp2_data['name'], linewidth=2.5, color='#4ECDC4')
    ax4.set_xlabel('Episode Return', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax4.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Rolling mean comparison
    ax5 = fig.add_subplot(gs[1, 2])
    window = max(3, min(5, len(exp1_data['returns']) // 10))

    if len(exp1_data['returns']) >= window:
        rolling_mean1 = pd.Series(exp1_data['returns']).rolling(window=window, center=True).mean()
        ax5.plot(exp1_data['steps'], rolling_mean1, label=f'{exp1_data["name"]} (rolling)',
                linewidth=2.5, color='#FF6B6B')

    if len(exp2_data['returns']) >= window:
        rolling_mean2 = pd.Series(exp2_data['returns']).rolling(window=window, center=True).mean()
        ax5.plot(exp2_data['steps'], rolling_mean2, label=f'{exp2_data["name"]} (rolling)',
                linewidth=2.5, color='#4ECDC4')

    ax5.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Rolling Mean Return', fontsize=11, fontweight='bold')
    ax5.set_title(f'Smoothed Performance (window={window})', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

    # Plot 6: Statistics table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    stats_data = [
        ['Metric', exp1_data['name'], exp2_data['name'], 'Difference'],
        ['Mean Return', f"{exp1_data['mean']:.2f}", f"{exp2_data['mean']:.2f}",
         f"{exp2_data['mean'] - exp1_data['mean']:+.2f}"],
        ['Std Dev', f"{exp1_data['std']:.2f}", f"{exp2_data['std']:.2f}",
         f"{exp2_data['std'] - exp1_data['std']:+.2f}"],
        ['Min Return', f"{exp1_data['min']:.2f}", f"{exp2_data['min']:.2f}",
         f"{exp2_data['min'] - exp1_data['min']:+.2f}"],
        ['Max Return', f"{exp1_data['max']:.2f}", f"{exp2_data['max']:.2f}",
         f"{exp2_data['max'] - exp1_data['max']:+.2f}"],
        ['Median', f"{exp1_data['median']:.2f}", f"{exp2_data['median']:.2f}",
         f"{exp2_data['median'] - exp1_data['median']:+.2f}"],
        ['Coef. of Var.', f"{exp1_data['metrics']['cv']:.3f}", f"{exp2_data['metrics']['cv']:.3f}",
         f"{exp2_data['metrics']['cv'] - exp1_data['metrics']['cv']:+.3f}"],
        ['Severity Score', f"{exp1_data['metrics']['severity_score']}/15",
         f"{exp2_data['metrics']['severity_score']}/15",
         f"{exp2_data['metrics']['severity_score'] - exp1_data['metrics']['severity_score']:+d}"],
        ['Policy Collapse?',
         '⚠ YES' if exp1_data['has_collapse'] else '✓ NO',
         '⚠ YES' if exp2_data['has_collapse'] else '✓ NO',
         '']
    ]

    table = ax6.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.22, 0.26, 0.26, 0.26])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

    # Style data rows
    for i in range(1, len(stats_data)):
        table[(i, 0)].set_facecolor('#E8E8E8')
        table[(i, 0)].set_text_props(weight='bold')
        for j in range(1, 4):
            if i == len(stats_data) - 1:  # Last row (collapse status)
                if 'YES' in stats_data[i][j]:
                    table[(i, j)].set_facecolor('#FFE5E5')
                elif 'NO' in stats_data[i][j]:
                    table[(i, j)].set_facecolor('#E5FFE5')
                else:
                    table[(i, j)].set_facecolor('#F5F5F5')
            else:
                table[(i, j)].set_facecolor('#F9F9F9')

    ax6.set_title('Detailed Performance Statistics', fontsize=14, fontweight='bold', pad=20)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Detailed comparison plot saved to: {save_path}")

    return fig

def main():
    print("="*80)
    print("Adam Optimizer vs L2-Regularized Learned Optimizer Comparison")
    print("="*80)

    # Define experiments
    exp1 = {
        'name': 'PPO + Adam (Baseline)',
        'path': './logs/ppo_ant_nonstationary_adam/1'
    }

    exp2 = {
        'name': 'PPO + Learned Optim4RL (L2 Reg)',
        'path': './logs/ppo_ant_nonstationary_test_learned_l2reg/1'
    }

    # Analyze both experiments
    exp1_data = analyze_experiment_detailed(exp1['name'], exp1['path'])
    exp2_data = analyze_experiment_detailed(exp2['name'], exp2['path'])

    if exp1_data is None or exp2_data is None:
        print("\n❌ Failed to load data for one or both experiments")
        return

    # Create detailed comparison plot
    create_detailed_comparison_plot(exp1_data, exp2_data,
                                   save_path='./results/adam_vs_l2reg_learned_comparison.png')

    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("🎯 COMPREHENSIVE COMPARISON SUMMARY")
    print(f"{'='*80}")

    print(f"\n📊 {exp1_data['name']}:")
    print(f"  Mean Return: {exp1_data['mean']:.2f} ± {exp1_data['std']:.2f}")
    print(f"  Median Return: {exp1_data['median']:.2f}")
    print(f"  Range: [{exp1_data['min']:.2f}, {exp1_data['max']:.2f}]")
    print(f"  Coefficient of Variation: {exp1_data['metrics']['cv']:.3f}")
    print(f"  Severity Score: {exp1_data['metrics']['severity_score']}/15")
    print(f"  Policy Collapse: {'⚠️  YES' if exp1_data['has_collapse'] else '✅ NO'}")

    print(f"\n📊 {exp2_data['name']}:")
    print(f"  Mean Return: {exp2_data['mean']:.2f} ± {exp2_data['std']:.2f}")
    print(f"  Median Return: {exp2_data['median']:.2f}")
    print(f"  Range: [{exp2_data['min']:.2f}, {exp2_data['max']:.2f}]")
    print(f"  Coefficient of Variation: {exp2_data['metrics']['cv']:.3f}")
    print(f"  Severity Score: {exp2_data['metrics']['severity_score']}/15")
    print(f"  Policy Collapse: {'⚠️  YES' if exp2_data['has_collapse'] else '✅ NO'}")

    # Performance comparison
    mean_diff = exp2_data['mean'] - exp1_data['mean']
    median_diff = exp2_data['median'] - exp1_data['median']

    print(f"\n🔍 Performance Difference:")
    print(f"  Mean difference: {mean_diff:+.2f}")
    print(f"  Median difference: {median_diff:+.2f}")

    if abs(mean_diff) < 100:
        print(f"  ➡️  Similar performance")
    elif mean_diff > 0:
        pct_improvement = (mean_diff / exp1_data['mean']) * 100
        print(f"  ✅ L2Reg performs better by {mean_diff:.2f} ({pct_improvement:+.1f}%)")
    else:
        pct_degradation = (abs(mean_diff) / exp1_data['mean']) * 100
        print(f"  ❌ L2Reg performs worse by {abs(mean_diff):.2f} ({-pct_degradation:.1f}%)")

    # Stability comparison
    print(f"\n📈 Stability Analysis:")
    if exp1_data['metrics']['cv'] < exp2_data['metrics']['cv']:
        print(f"  ✅ {exp1_data['name']} is more stable (CV: {exp1_data['metrics']['cv']:.3f} vs {exp2_data['metrics']['cv']:.3f})")
    else:
        print(f"  ✅ {exp2_data['name']} is more stable (CV: {exp2_data['metrics']['cv']:.3f} vs {exp1_data['metrics']['cv']:.3f})")

    # Recommendation
    print(f"\n💡 Recommendation:")
    if exp1_data['has_collapse'] and not exp2_data['has_collapse']:
        print(f"  ✅ Use {exp2_data['name']} - no collapse detected")
    elif not exp1_data['has_collapse'] and exp2_data['has_collapse']:
        print(f"  ✅ Use {exp1_data['name']} - no collapse detected")
    elif not exp1_data['has_collapse'] and not exp2_data['has_collapse']:
        if mean_diff > 100:
            print(f"  ✅ Use {exp2_data['name']} - better performance, no collapse")
        elif mean_diff < -100:
            print(f"  ✅ Use {exp1_data['name']} - better performance, no collapse")
        else:
            if exp1_data['metrics']['cv'] < exp2_data['metrics']['cv']:
                print(f"  ✅ Use {exp1_data['name']} - similar performance but more stable")
            else:
                print(f"  ✅ Use {exp2_data['name']} - similar performance but more stable")
    else:
        print(f"  ⚠️  Both show collapse - further investigation needed")

    print(f"\n{'='*80}")

if __name__ == '__main__':
    main()
