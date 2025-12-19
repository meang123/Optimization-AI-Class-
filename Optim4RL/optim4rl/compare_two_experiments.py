#!/usr/bin/env python3
"""
Compare two experiment results and detect policy collapse
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

def load_pickle_inference(path):
    """Load inference pickle file"""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def detect_policy_collapse(returns, threshold_percentile=10, window_size=5):
    """
    Detect policy collapse by checking for:
    1. Sudden drop in returns
    2. Sustained low performance
    3. High variance (unstable policy)
    """
    if len(returns) < window_size:
        return False, "Not enough data"

    # Calculate statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)

    # Check for collapse indicators
    collapse_indicators = []

    # 1. Very low returns (bottom 10%)
    threshold = np.percentile(returns, threshold_percentile)
    low_return_ratio = np.sum(returns < threshold) / len(returns)
    if low_return_ratio > 0.3:
        collapse_indicators.append(f"High ratio of low returns: {low_return_ratio:.2%}")

    # 2. Negative or very low mean return
    if mean_return < -100:
        collapse_indicators.append(f"Very low mean return: {mean_return:.2f}")

    # 3. Large standard deviation (unstable)
    if std_return > abs(mean_return) * 0.5 and std_return > 200:
        collapse_indicators.append(f"High variance: std={std_return:.2f}")

    # 4. Sudden drops (compare consecutive windows)
    for i in range(window_size, len(returns)):
        prev_window = returns[i-window_size:i]
        curr_window = returns[max(0, i-window_size):i+1]

        if np.mean(prev_window) - np.mean(curr_window) > 500:
            collapse_indicators.append(f"Sudden drop at step {i}")
            break

    # 5. Check if most returns are below a reasonable threshold
    if np.median(returns) < -200:
        collapse_indicators.append(f"Very low median return: {np.median(returns):.2f}")

    has_collapse = len(collapse_indicators) > 0

    return has_collapse, collapse_indicators

def analyze_experiment(exp_name, log_path):
    """Analyze a single experiment"""
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

    print(f"\n📊 Data Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLast few rows:")
    print(df.tail())

    # Extract returns
    returns = df['Return'].values
    steps = df['Step'].values

    # Statistics
    print(f"\n📈 Statistics:")
    print(f"  Total evaluations: {len(returns)}")
    print(f"  Mean return: {np.mean(returns):.2f}")
    print(f"  Std return: {np.std(returns):.2f}")
    print(f"  Min return: {np.min(returns):.2f}")
    print(f"  Max return: {np.max(returns):.2f}")
    print(f"  Median return: {np.median(returns):.2f}")
    print(f"  25th percentile: {np.percentile(returns, 25):.2f}")
    print(f"  75th percentile: {np.percentile(returns, 75):.2f}")

    # Detect collapse
    has_collapse, indicators = detect_policy_collapse(returns)

    print(f"\n🔍 Policy Collapse Detection:")
    if has_collapse:
        print(f"  ⚠️  POLICY COLLAPSE DETECTED!")
        print(f"  Indicators:")
        for indicator in indicators:
            print(f"    - {indicator}")
    else:
        print(f"  ✅ No clear policy collapse detected")

    # Check for inference pickle
    inference_pickle_path = os.path.join(log_path, "inference_param_iter5.pickle")
    if os.path.exists(inference_pickle_path):
        print(f"\n📦 Inference pickle found: {inference_pickle_path}")
        inf_data = load_pickle_inference(inference_pickle_path)
        if inf_data is not None:
            print(f"  Pickle type: {type(inf_data)}")
            if isinstance(inf_data, dict):
                print(f"  Keys: {list(inf_data.keys())}")

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
        'collapse_indicators': indicators
    }

def create_comparison_plot(exp1_data, exp2_data, save_path='comparison_plot.png'):
    """Create comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Returns over steps
    ax1 = axes[0, 0]
    ax1.plot(exp1_data['steps'], exp1_data['returns'],
             label=exp1_data['name'], marker='o', markersize=4, alpha=0.7, linewidth=2)
    ax1.plot(exp2_data['steps'], exp2_data['returns'],
             label=exp2_data['name'], marker='s', markersize=4, alpha=0.7, linewidth=2)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Episode Return', fontsize=12)
    ax1.set_title('Episode Returns Over Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Box plot comparison
    ax2 = axes[0, 1]
    box_data = [exp1_data['returns'], exp2_data['returns']]
    bp = ax2.boxplot(box_data, labels=[exp1_data['name'], exp2_data['name']], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
        patch.set_facecolor(color)
    ax2.set_ylabel('Episode Return', fontsize=12)
    ax2.set_title('Return Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Histogram comparison
    ax3 = axes[1, 0]
    ax3.hist(exp1_data['returns'], bins=20, alpha=0.6, label=exp1_data['name'], color='blue')
    ax3.hist(exp2_data['returns'], bins=20, alpha=0.6, label=exp2_data['name'], color='green')
    ax3.set_xlabel('Episode Return', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Return Distribution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Statistics comparison table
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_data = [
        ['Metric', exp1_data['name'], exp2_data['name']],
        ['Mean', f"{exp1_data['mean']:.2f}", f"{exp2_data['mean']:.2f}"],
        ['Std', f"{exp1_data['std']:.2f}", f"{exp2_data['std']:.2f}"],
        ['Min', f"{exp1_data['min']:.2f}", f"{exp2_data['min']:.2f}"],
        ['Max', f"{exp1_data['max']:.2f}", f"{exp2_data['max']:.2f}"],
        ['Median', f"{exp1_data['median']:.2f}", f"{exp2_data['median']:.2f}"],
        ['Collapse?',
         '⚠️  YES' if exp1_data['has_collapse'] else '✅ NO',
         '⚠️  YES' if exp2_data['has_collapse'] else '✅ NO']
    ]

    table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(stats_data)):
        for j in range(3):
            if j == 0:
                table[(i, j)].set_facecolor('#E8E8E8')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')

    ax4.set_title('Performance Statistics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {save_path}")

    return fig

def main():
    # Define experiments to compare
    exp1 = {
        'name': 'PPO-Ant (Learned Optim4RL)',
        'path': './logs/ppo_ant_non_tested/1'
    }

    # Note: MetaPPO is meta-learning, so we compare with Adam baseline instead
    exp2 = {
        'name': 'PPO-Ant (Adam Baseline)',
        'path': './logs/ppo_ant_nonstationary_adam/1'
    }

    # Analyze both experiments
    exp1_data = analyze_experiment(exp1['name'], exp1['path'])
    exp2_data = analyze_experiment(exp2['name'], exp2['path'])

    # Check if we have data for both
    if exp1_data is None:
        print(f"\n❌ Failed to load data for {exp1['name']}")
        print(f"Note: This might be because it's a standard RL experiment with test results.")
        return

    if exp2_data is None:
        print(f"\n❌ Failed to load data for {exp2['name']}")
        return

    if exp1_data and exp2_data:
        # Create comparison plot
        create_comparison_plot(exp1_data, exp2_data,
                              save_path='./results/two_experiments_comparison.png')

        # Print summary
        print(f"\n{'='*80}")
        print("📊 COMPARISON SUMMARY")
        print(f"{'='*80}")

        print(f"\n{exp1_data['name']}:")
        print(f"  Mean Return: {exp1_data['mean']:.2f} ± {exp1_data['std']:.2f}")
        print(f"  Policy Collapse: {'⚠️  YES' if exp1_data['has_collapse'] else '✅ NO'}")

        print(f"\n{exp2_data['name']}:")
        print(f"  Mean Return: {exp2_data['mean']:.2f} ± {exp2_data['std']:.2f}")
        print(f"  Policy Collapse: {'⚠️  YES' if exp2_data['has_collapse'] else '✅ NO'}")

        # Performance comparison
        mean_diff = exp1_data['mean'] - exp2_data['mean']
        print(f"\n🎯 Performance Difference:")
        if abs(mean_diff) < 50:
            print(f"  Similar performance (diff: {mean_diff:.2f})")
        elif mean_diff > 0:
            print(f"  {exp1_data['name']} performs better (+{mean_diff:.2f})")
        else:
            print(f"  {exp2_data['name']} performs better (+{abs(mean_diff):.2f})")

if __name__ == '__main__':
    main()
