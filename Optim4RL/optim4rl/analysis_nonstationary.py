# Copyright 2024 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Analysis script for Non-Stationary Task Results

This script analyzes training results from PPO and MetaPPO on non-stationary tasks,
focusing on policy collapse detection and robustness comparison.

Usage:
    python analysis_nonstationary.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
from utils.collapse_detector import (
    detect_collapse,
    compute_stability_score,
    analyze_collapse_recovery,
    compare_collapse_rates,
    generate_collapse_summary
)

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_results(exp_name: str, config_idx: int = 1) -> pd.DataFrame:
    """
    Load training results from logs directory.

    Args:
        exp_name: Experiment name (e.g., 'ppo_ant_nonstationary')
        config_idx: Configuration index

    Returns:
        DataFrame with training results
    """
    log_path = f'./logs/{exp_name}/{config_idx}/result_Test.feather'

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Results not found at {log_path}")

    df = pd.read_feather(log_path)
    print(f"Loaded {len(df)} episodes from {log_path}")

    return df


def plot_training_curve_with_collapse(
    df: pd.DataFrame,
    collapse_events: List[Dict],
    title: str,
    save_path: str = None
):
    """
    Plot training curve with collapse events highlighted.

    Args:
        df: Training results DataFrame
        collapse_events: List of collapse events
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot training curve
    ax.plot(df['Step'], df['Return'], linewidth=1.5, alpha=0.7, label='Episode Return')

    # Plot rolling mean
    window = 100
    rolling_mean = df['Return'].rolling(window=window, min_periods=1).mean()
    ax.plot(df['Step'], rolling_mean, linewidth=2, color='red', label=f'Rolling Mean ({window} eps)')

    # Highlight collapse events
    if collapse_events:
        collapse_steps = [df.iloc[e['step']]['Step'] for e in collapse_events]
        collapse_returns = [e['collapsed_value'] for e in collapse_events]
        ax.scatter(collapse_steps, collapse_returns, color='red', s=100, marker='x',
                   linewidths=3, label=f'Collapse Events ({len(collapse_events)})', zorder=5)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Episode Return', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_collapse_comparison(
    ppo_events: List[Dict],
    metappo_events: List[Dict],
    save_path: str = None
):
    """
    Plot comparison of collapse events between PPO and MetaPPO.

    Args:
        ppo_events: PPO collapse events
        metappo_events: MetaPPO collapse events
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Collapse count comparison
    counts = [len(ppo_events), len(metappo_events)]
    labels = ['PPO', 'MetaPPO']
    colors = ['#e74c3c', '#3498db']

    axes[0].bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Number of Collapses', fontsize=12)
    axes[0].set_title('Collapse Frequency', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for i, count in enumerate(counts):
        axes[0].text(i, count + max(counts) * 0.02, str(count),
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Collapse severity distribution
    if ppo_events and metappo_events:
        ppo_drops = [e['drop_percentage'] for e in ppo_events]
        metappo_drops = [e['drop_percentage'] for e in metappo_events]

        axes[1].hist([ppo_drops, metappo_drops], bins=20, label=labels,
                     color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Performance Drop (%)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Collapse Severity Distribution', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                     transform=axes[1].transAxes, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    plt.show()


def generate_comparison_report(
    ppo_df: pd.DataFrame,
    metappo_df: pd.DataFrame,
    task_name: str = 'ant'
) -> Dict:
    """
    Generate comprehensive comparison report between PPO and MetaPPO.

    Args:
        ppo_df: PPO training results
        metappo_df: MetaPPO training results
        task_name: Task name (ant or humanoid)

    Returns:
        Dictionary with comparison metrics
    """
    print(f"\n{'='*80}")
    print(f"POLICY COLLAPSE ANALYSIS REPORT: {task_name.upper()}")
    print(f"{'='*80}\n")

    # Extract rewards
    ppo_rewards = ppo_df['Return'].values
    metappo_rewards = metappo_df['Return'].values

    # Detect collapses
    ppo_collapses = detect_collapse(ppo_rewards, window_size=100, threshold=0.5)
    metappo_collapses = detect_collapse(metappo_rewards, window_size=100, threshold=0.5)

    # Compute stability scores
    ppo_stability = compute_stability_score(ppo_rewards)
    metappo_stability = compute_stability_score(metappo_rewards)

    # Analyze recovery
    ppo_recovery = analyze_collapse_recovery(ppo_rewards, ppo_collapses)
    metappo_recovery = analyze_collapse_recovery(metappo_rewards, metappo_collapses)

    # Generate summaries
    ppo_summary = generate_collapse_summary(ppo_collapses)
    metappo_summary = generate_collapse_summary(metappo_collapses)

    # Print report
    print("1. COLLAPSE FREQUENCY")
    print("-" * 80)
    print(f"PPO Collapses:     {ppo_summary['total_collapses']}")
    print(f"MetaPPO Collapses: {metappo_summary['total_collapses']}")
    print(f"Improvement:       {ppo_summary['total_collapses'] - metappo_summary['total_collapses']} fewer collapses")
    if ppo_summary['total_collapses'] > 0:
        improvement_pct = (1 - metappo_summary['total_collapses'] / ppo_summary['total_collapses']) * 100
        print(f"                   {improvement_pct:.1f}% reduction\n")

    print("2. COLLAPSE SEVERITY")
    print("-" * 80)
    print(f"PPO Mean Drop:     {ppo_summary['mean_drop']:.2f}%")
    print(f"PPO Max Drop:      {ppo_summary['max_drop']:.2f}%")
    print(f"MetaPPO Mean Drop: {metappo_summary['mean_drop']:.2f}%")
    print(f"MetaPPO Max Drop:  {metappo_summary['max_drop']:.2f}%\n")

    print("3. PERFORMANCE STABILITY")
    print("-" * 80)
    print(f"PPO Stability Score:     {ppo_stability:.4f}")
    print(f"MetaPPO Stability Score: {metappo_stability:.4f}")
    stability_improvement = (1 - metappo_stability / ppo_stability) * 100 if ppo_stability > 0 else 0
    print(f"Improvement:             {stability_improvement:.1f}% more stable\n")

    print("4. RECOVERY ANALYSIS")
    print("-" * 80)
    if ppo_recovery:
        ppo_recovered = sum(1 for r in ppo_recovery if r['recovered'])
        ppo_recovery_rate = ppo_recovered / len(ppo_recovery) * 100
        ppo_avg_recovery_time = np.mean([r['recovery_time'] for r in ppo_recovery if r['recovery_time'] is not None])
        print(f"PPO Recovery Rate:     {ppo_recovery_rate:.1f}% ({ppo_recovered}/{len(ppo_recovery)})")
        print(f"PPO Avg Recovery Time: {ppo_avg_recovery_time:.0f} episodes" if not np.isnan(ppo_avg_recovery_time) else "PPO Avg Recovery Time: N/A")
    else:
        print("PPO Recovery Rate:     N/A (no collapses)")

    if metappo_recovery:
        metappo_recovered = sum(1 for r in metappo_recovery if r['recovered'])
        metappo_recovery_rate = metappo_recovered / len(metappo_recovery) * 100
        metappo_avg_recovery_time = np.mean([r['recovery_time'] for r in metappo_recovery if r['recovery_time'] is not None])
        print(f"MetaPPO Recovery Rate:     {metappo_recovery_rate:.1f}% ({metappo_recovered}/{len(metappo_recovery)})")
        print(f"MetaPPO Avg Recovery Time: {metappo_avg_recovery_time:.0f} episodes" if not np.isnan(metappo_avg_recovery_time) else "MetaPPO Avg Recovery Time: N/A")
    else:
        print("MetaPPO Recovery Rate:     N/A (no collapses)")

    print("\n5. OVERALL PERFORMANCE")
    print("-" * 80)
    print(f"PPO Final Return (last 100 eps):     {ppo_rewards[-100:].mean():.2f} ± {ppo_rewards[-100:].std():.2f}")
    print(f"MetaPPO Final Return (last 100 eps): {metappo_rewards[-100:].mean():.2f} ± {metappo_rewards[-100:].std():.2f}")

    print(f"\n{'='*80}\n")

    return {
        'ppo_collapses': ppo_collapses,
        'metappo_collapses': metappo_collapses,
        'ppo_summary': ppo_summary,
        'metappo_summary': metappo_summary,
        'ppo_stability': ppo_stability,
        'metappo_stability': metappo_stability,
        'ppo_recovery': ppo_recovery,
        'metappo_recovery': metappo_recovery
    }


def main():
    """
    Main analysis pipeline for non-stationary tasks.
    """
    # Create results directory
    results_dir = Path('./results/nonstationary_analysis')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Analyze Ant task
    print("\n" + "="*80)
    print("ANALYZING ANT NON-STATIONARY TASK")
    print("="*80)

    try:
        ppo_ant = load_results('ppo_ant_nonstationary', config_idx=1)
        metappo_ant = load_results('metap_ppo_ant_nonstationary', config_idx=1)

        # Generate report
        ant_report = generate_comparison_report(ppo_ant, metappo_ant, task_name='ant')

        # Plot PPO results
        plot_training_curve_with_collapse(
            ppo_ant,
            ant_report['ppo_collapses'],
            'PPO Ant Non-Stationary Training',
            save_path=str(results_dir / 'ppo_ant_training.png')
        )

        # Plot MetaPPO results
        plot_training_curve_with_collapse(
            metappo_ant,
            ant_report['metappo_collapses'],
            'MetaPPO Ant Non-Stationary Training',
            save_path=str(results_dir / 'metappo_ant_training.png')
        )

        # Plot comparison
        plot_collapse_comparison(
            ant_report['ppo_collapses'],
            ant_report['metappo_collapses'],
            save_path=str(results_dir / 'ant_collapse_comparison.png')
        )

    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Skipping Ant analysis - results not found")

    # Analyze Humanoid task
    print("\n" + "="*80)
    print("ANALYZING HUMANOID NON-STATIONARY TASK")
    print("="*80)

    try:
        ppo_humanoid = load_results('ppo_humanoid_nonstationary', config_idx=1)
        metappo_humanoid = load_results('metap_ppo_humanoid_nonstationary', config_idx=1)

        # Generate report
        humanoid_report = generate_comparison_report(ppo_humanoid, metappo_humanoid, task_name='humanoid')

        # Plot PPO results
        plot_training_curve_with_collapse(
            ppo_humanoid,
            humanoid_report['ppo_collapses'],
            'PPO Humanoid Non-Stationary Training',
            save_path=str(results_dir / 'ppo_humanoid_training.png')
        )

        # Plot MetaPPO results
        plot_training_curve_with_collapse(
            metappo_humanoid,
            humanoid_report['metappo_collapses'],
            'MetaPPO Humanoid Non-Stationary Training',
            save_path=str(results_dir / 'metappo_humanoid_training.png')
        )

        # Plot comparison
        plot_collapse_comparison(
            humanoid_report['ppo_collapses'],
            humanoid_report['metappo_collapses'],
            save_path=str(results_dir / 'humanoid_collapse_comparison.png')
        )

    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Skipping Humanoid analysis - results not found")

    print(f"\nAnalysis complete! Results saved to {results_dir}/")


if __name__ == '__main__':
    main()
