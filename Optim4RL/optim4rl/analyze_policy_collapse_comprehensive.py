#!/usr/bin/env python3
"""Comprehensive analysis of policy collapse across different optimizers and configurations."""

import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

def parse_log_file(log_path):
    """Parse log file and extract iteration and return values."""
    returns = []
    iterations = []
    steps = []

    with open(log_path, 'r') as f:
        for line in f:
            # Match pattern: Iteration X/Y, Step Z, Return=W
            match = re.search(r'Iteration (\d+)/\d+.*?Step (\d+).*?Return=([-\d.]+)', line)
            if match:
                iteration = int(match.group(1))
                step = int(match.group(2))
                return_val = float(match.group(3))
                iterations.append(iteration)
                steps.append(step)
                returns.append(return_val)

    return np.array(iterations), np.array(steps), np.array(returns)

def detect_collapse(returns, threshold=0.5):
    """Detect policy collapse based on performance degradation."""
    if len(returns) < 2:
        return False, 0.0, {}

    # Calculate metrics
    max_return = np.max(returns)
    min_return = np.min(returns)
    mean_return = np.mean(returns)
    final_10_mean = np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns[-len(returns)//2:])

    # Check for collapse conditions
    collapse_detected = False
    collapse_type = []

    # 1. Final performance is negative while max was positive
    if max_return > 0 and final_10_mean < 0:
        collapse_detected = True
        collapse_type.append("negative_final")

    # 2. Severe performance drop (>50% from peak)
    if max_return > 0:
        drop_percentage = (max_return - final_10_mean) / max_return
        if drop_percentage > threshold:
            collapse_detected = True
            collapse_type.append(f"severe_drop_{drop_percentage*100:.1f}%")

    # 3. Negative mean performance
    if mean_return < 0:
        collapse_detected = True
        collapse_type.append("negative_mean")

    # 4. High variance with declining trend
    std_return = np.std(returns)
    cv = abs(std_return / mean_return) if mean_return != 0 else 0

    metrics = {
        'max': max_return,
        'min': min_return,
        'mean': mean_return,
        'std': std_return,
        'cv': cv,
        'final_10_mean': final_10_mean,
        'drop_from_peak': (max_return - final_10_mean) / max_return if max_return > 0 else 0,
        'collapse_type': collapse_type
    }

    return collapse_detected, drop_percentage if max_return > 0 else 0, metrics

def analyze_experiment(name, log_path, config_path):
    """Analyze a single experiment."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {name}")
    print(f"{'='*80}")

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Parse log
    iterations, steps, returns = parse_log_file(log_path)

    if len(returns) == 0:
        print(f"WARNING: No data found in {log_path}")
        return None

    # Detect collapse
    collapsed, drop_pct, metrics = detect_collapse(returns)

    # Print config info
    print(f"\nConfiguration:")
    if 'optim' in config:
        optim_name = config['optim'][0]['name'][0]
        optim_kwargs = config['optim'][0].get('kwargs', [{}])[0]
        print(f"  Optimizer: {optim_name}")
        print(f"  Optimizer kwargs: {optim_kwargs}")
    elif 'agent_optim' in config:
        optim_name = config['agent_optim'][0]['name'][0]
        optim_kwargs = config['agent_optim'][0].get('kwargs', [{}])[0]
        print(f"  Agent Optimizer: {optim_name}")
        print(f"  Agent Optimizer kwargs: {optim_kwargs}")
        if 'meta_optim' in config:
            meta_name = config['meta_optim'][0]['name'][0]
            print(f"  Meta Optimizer: {meta_name}")

    agent_name = config['agent'][0]['name'][0]
    print(f"  Agent: {agent_name}")

    if 'l2_weight' in config['agent'][0]:
        l2_weight = config['agent'][0].get('l2_weight', [0.0])[0]
        print(f"  L2 Weight: {l2_weight}")

    # Print results
    print(f"\nResults ({len(returns)} iterations):")
    print(f"  Max Return: {metrics['max']:.2f}")
    print(f"  Min Return: {metrics['min']:.2f}")
    print(f"  Mean Return: {metrics['mean']:.2f} ± {metrics['std']:.2f}")
    print(f"  Final 10 Mean: {metrics['final_10_mean']:.2f}")
    print(f"  Coefficient of Variation: {metrics['cv']:.3f}")
    print(f"  Drop from Peak: {metrics['drop_from_peak']*100:.1f}%")

    print(f"\nPolicy Collapse Analysis:")
    if collapsed:
        print(f"  ⚠️  COLLAPSE DETECTED")
        print(f"  Collapse Types: {', '.join(metrics['collapse_type'])}")
    else:
        print(f"  ✅ NO COLLAPSE - Training Stable")

    return {
        'name': name,
        'config': config,
        'iterations': iterations,
        'steps': steps,
        'returns': returns,
        'collapsed': collapsed,
        'metrics': metrics
    }

def compare_experiments(results):
    """Compare multiple experiments."""
    print(f"\n\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}")

    # Create comparison table
    print(f"\n{'Experiment':<50} {'Mean':<12} {'Final10':<12} {'Collapsed':<12}")
    print("-" * 86)
    for result in results:
        if result:
            name = result['name'][:48]
            mean = result['metrics']['mean']
            final10 = result['metrics']['final_10_mean']
            collapsed = "YES ⚠️" if result['collapsed'] else "NO ✅"
            print(f"{name:<50} {mean:>10.2f}  {final10:>10.2f}  {collapsed:<12}")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Policy Collapse Analysis: Comprehensive Comparison', fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, result in enumerate(results):
        if result is None:
            continue

        ax = axes[idx // 3, idx % 3]
        returns = result['returns']
        iterations = result['iterations']

        # Plot returns
        ax.plot(iterations, returns, 'o-', color=colors[idx], linewidth=2, markersize=4, alpha=0.7)

        # Add horizontal lines for mean and final
        ax.axhline(result['metrics']['mean'], color=colors[idx], linestyle='--', alpha=0.5, label='Mean')
        ax.axhline(result['metrics']['final_10_mean'], color='red', linestyle='--', alpha=0.5, label='Final 10 Mean')

        # Mark max and min
        max_idx = np.argmax(returns)
        min_idx = np.argmin(returns)
        ax.scatter(iterations[max_idx], returns[max_idx], color='green', s=100, marker='^', zorder=5, label='Max')
        ax.scatter(iterations[min_idx], returns[min_idx], color='red', s=100, marker='v', zorder=5, label='Min')

        # Styling
        title = result['name']
        if result['collapsed']:
            title += " ⚠️ COLLAPSED"
            ax.set_facecolor('#fff0f0')
        else:
            title += " ✅ STABLE"
            ax.set_facecolor('#f0fff0')

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Return')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Add text box with metrics
        textstr = f'Mean: {result["metrics"]["mean"]:.1f}\nFinal: {result["metrics"]["final_10_mean"]:.1f}\nDrop: {result["metrics"]["drop_from_peak"]*100:.1f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)

    # Hide the last unused subplot
    if len(results) < 6:
        axes[1, 2].set_visible(False)

    plt.tight_layout()
    plt.savefig('results/comprehensive_collapse_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: results/comprehensive_collapse_analysis.png")

    return fig

def main():
    # Define experiments
    experiments = [
        {
            'name': '1. PPO + Adam (default β)',
            'log': 'logs/ppo_ant_nonstationary/1/log.txt',
            'config': 'configs/ppo_ant_nonstationary.json'
        },
        {
            'name': '2. PPO + Adam (β1=β2=0.997)',
            'log': 'logs/ppo_ant_nonstationary_adam/1/log.txt',
            'config': 'configs/ppo_ant_nonstationary_adam.json'
        },
        {
            'name': '3. PPO + Learned Optim4RL (no L2)',
            'log': 'logs/ppo_ant_non_tested/1/log.txt',
            'config': 'configs/ppo_ant_non_tested.json'
        },
        {
            'name': '4. PPO + Learned Optim4RL + L2',
            'log': 'logs/ppo_ant_nonstationary_test_learned_l2reg/1/log.txt',
            'config': 'configs/ppo_ant_nonstationary_test_learned_l2reg.json'
        },
        {
            'name': '5. PPO + Optim4RL (stationary→nonstat)',
            'log': 'logs/stationary_optim/1/log.txt',
            'config': 'configs/stationary_optim.json'
        }
    ]

    # Analyze each experiment
    results = []
    for exp in experiments:
        result = analyze_experiment(exp['name'], exp['log'], exp['config'])
        results.append(result)

    # Compare all experiments
    compare_experiments(results)

    # Generate final report
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    print("\n🔍 Key Findings:")

    # Find which ones collapsed
    collapsed_exps = [r['name'] for r in results if r and r['collapsed']]
    stable_exps = [r['name'] for r in results if r and not r['collapsed']]

    print(f"\n✅ Stable Experiments ({len(stable_exps)}):")
    for name in stable_exps:
        print(f"  - {name}")

    print(f"\n⚠️  Collapsed Experiments ({len(collapsed_exps)}):")
    for name in collapsed_exps:
        print(f"  - {name}")

    # Performance ranking
    print("\n📊 Performance Ranking (by mean return):")
    sorted_results = sorted([r for r in results if r], key=lambda x: x['metrics']['mean'], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        print(f"  {i}. {result['name']}: {result['metrics']['mean']:.2f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
