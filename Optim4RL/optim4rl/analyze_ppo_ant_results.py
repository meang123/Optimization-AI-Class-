#!/usr/bin/env python3
"""
Quick analysis script for PPO Ant Non-Stationary results
Based on Task.md Phase 4 requirements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.collapse_detector import (
    detect_collapse,
    compute_stability_score,
    analyze_collapse_recovery,
    generate_collapse_summary
)

# Load results
print("="*80)
print("PPO ANT NON-STATIONARY RESULTS ANALYSIS")
print("="*80)
print()

results_path = './logs/ppo_ant_nonstationary/1/result_Test.feather'
df = pd.read_feather(results_path)

print(f"✓ Loaded {len(df)} evaluation episodes")
print(f"✓ Training steps: {df['Step'].min():.0f} to {df['Step'].max():.0f}")
print(f"✓ Total training: {df['Step'].max()/1e9:.2f} billion steps")
print()

# Extract rewards
rewards = df['Return'].values

# Basic statistics
print("="*80)
print("1. BASIC PERFORMANCE METRICS")
print("="*80)
print(f"Mean Return:     {rewards.mean():.2f}")
print(f"Std Dev:         {rewards.std():.2f}")
print(f"Min Return:      {rewards.min():.2f}")
print(f"Max Return:      {rewards.max():.2f}")
print(f"First 5 evals:   {rewards[:5].mean():.2f}")
print(f"Last 5 evals:    {rewards[-5:].mean():.2f}")
print(f"Performance drop: {((rewards[:5].mean() - rewards[-5:].mean()) / rewards[:5].mean() * 100):.1f}%")
print()

# Policy collapse detection
print("="*80)
print("2. POLICY COLLAPSE ANALYSIS")
print("="*80)

collapse_events = detect_collapse(rewards, window_size=5, threshold=0.5)
summary = generate_collapse_summary(collapse_events)

print(f"Total Collapses Detected: {summary['total_collapses']}")
if summary['total_collapses'] > 0:
    print(f"Mean Drop Severity:       {summary['mean_drop']:.2f}%")
    print(f"Max Drop Severity:        {summary['max_drop']:.2f}%")
    print(f"First Collapse at:        Episode {collapse_events[0]['step']}")
    print()
    print("Collapse Events:")
    for i, event in enumerate(collapse_events):
        print(f"  {i+1}. Episode {event['step']}: "
              f"{event['baseline_value']:.0f} → {event['collapsed_value']:.0f} "
              f"({event['drop_percentage']:.1f}% drop)")
else:
    print("No collapses detected (>50% drop)")
print()

# Stability analysis
print("="*80)
print("3. PERFORMANCE STABILITY")
print("="*80)

stability = compute_stability_score(rewards)
print(f"Stability Score: {stability:.4f}")
print(f"  (Lower is better, 0 = perfectly stable)")
print()

# Rolling statistics
window = 5
rolling_mean = pd.Series(rewards).rolling(window=window).mean()
rolling_std = pd.Series(rewards).rolling(window=window).std()

print(f"Rolling Mean (window={window}):")
print(f"  Early training (eps 5-10):  {rolling_mean[5:10].mean():.2f}")
print(f"  Mid training (eps 10-15):   {rolling_mean[10:15].mean():.2f}")
print(f"  Late training (eps 15-20):  {rolling_mean[15:20].mean():.2f}")
print()

# Recovery analysis
print("="*80)
print("4. RECOVERY ANALYSIS")
print("="*80)

recovery_info = analyze_collapse_recovery(rewards, collapse_events)
if recovery_info:
    recovered = sum(1 for r in recovery_info if r['recovered'])
    recovery_rate = recovered / len(recovery_info) * 100
    print(f"Recovery Rate: {recovery_rate:.1f}% ({recovered}/{len(recovery_info)})")

    recovery_times = [r['recovery_time'] for r in recovery_info if r['recovery_time'] is not None]
    if recovery_times:
        print(f"Average Recovery Time: {np.mean(recovery_times):.1f} episodes")
        print(f"Median Recovery Time:  {np.median(recovery_times):.1f} episodes")
else:
    print("No collapses to analyze recovery")
print()

# Detailed episode-by-episode view
print("="*80)
print("5. EPISODE-BY-EPISODE PERFORMANCE")
print("="*80)
print("Iteration | Training Steps | Episode Return | Status")
print("-"*80)
for idx, row in df.iterrows():
    status = ""
    if idx > 0:
        prev_return = df.iloc[idx-1]['Return']
        change = ((row['Return'] - prev_return) / prev_return * 100)
        if change < -50:
            status = "⚠️  COLLAPSE"
        elif change < -20:
            status = "⬇️  DROP"
        elif change > 20:
            status = "⬆️  RECOVERY"

    print(f"{idx:9d} | {row['Step']:14.0f} | {row['Return']:14.2f} | {status}")
print()

# Create visualization
print("="*80)
print("6. GENERATING VISUALIZATIONS")
print("="*80)

results_dir = Path('./results/ppo_ant_analysis')
results_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Training curve with collapses
ax1 = axes[0]
ax1.plot(df['Step']/1e6, df['Return'], 'o-', linewidth=2, markersize=8,
         label='Episode Return', color='#3498db')

# Highlight collapse events
if collapse_events:
    collapse_steps = [df.iloc[e['step']]['Step']/1e6 for e in collapse_events]
    collapse_returns = [e['collapsed_value'] for e in collapse_events]
    ax1.scatter(collapse_steps, collapse_returns, color='red', s=200, marker='X',
               linewidths=2, label=f'Collapse Events ({len(collapse_events)})',
               zorder=5, edgecolors='darkred')

ax1.set_xlabel('Training Steps (Millions)', fontsize=12)
ax1.set_ylabel('Episode Return', fontsize=12)
ax1.set_title('PPO Ant Non-Stationary Training - Performance Over Time',
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Add annotations for key events
if collapse_events:
    for event in collapse_events:
        step = df.iloc[event['step']]['Step']/1e6
        value = event['collapsed_value']
        ax1.annotate(f"-{event['drop_percentage']:.0f}%",
                    xy=(step, value), xytext=(10, -20),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Plot 2: Rolling mean and std
ax2 = axes[1]
episodes = np.arange(len(df))
ax2.plot(df['Step']/1e6, rolling_mean, linewidth=2, color='#2ecc71',
        label=f'Rolling Mean (window={window})')
ax2.fill_between(df['Step']/1e6,
                  rolling_mean - rolling_std,
                  rolling_mean + rolling_std,
                  alpha=0.3, color='#2ecc71', label='± 1 Std Dev')

ax2.set_xlabel('Training Steps (Millions)', fontsize=12)
ax2.set_ylabel('Episode Return', fontsize=12)
ax2.set_title('PPO Ant Non-Stationary - Performance Stability',
             fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_path = results_dir / 'ppo_ant_nonstationary_analysis.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to {save_path}")

# Create collapse timeline plot
if collapse_events:
    fig2, ax = plt.subplots(figsize=(14, 6))

    # Plot baseline and collapses
    for i, event in enumerate(collapse_events):
        step = df.iloc[event['step']]['Step']/1e6
        ax.axvline(x=step, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.text(step, ax.get_ylim()[1]*0.9, f"C{i+1}", ha='center',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    ax.plot(df['Step']/1e6, df['Return'], 'o-', linewidth=2, markersize=8, color='#3498db')
    ax.set_xlabel('Training Steps (Millions)', fontsize=12)
    ax.set_ylabel('Episode Return', fontsize=12)
    ax.set_title('PPO Ant Non-Stationary - Collapse Timeline',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    timeline_path = results_dir / 'ppo_ant_collapse_timeline.png'
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved collapse timeline to {timeline_path}")

print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Training completed: {df['Step'].max()/1e9:.2f}B / 1.0B steps ({df['Step'].max()/1e9*100:.1f}%)")
print(f"✓ Policy collapses detected: {summary['total_collapses']}")
if summary['total_collapses'] > 0:
    print(f"✓ Collapse rate: {summary['total_collapses']/len(df)*100:.1f}% of evaluations")
    print(f"⚠️  WARNING: Policy collapse observed in non-stationary environment!")
else:
    print(f"✓ No significant collapses detected")
print(f"✓ Performance stability score: {stability:.4f}")
print(f"✓ Results saved to: {results_dir}/")
print()
print("Next steps (from Task.md):")
print("  1. Wait for MetaPPO training to complete")
print("  2. Compare PPO vs MetaPPO collapse resistance")
print("  3. Run full comparison analysis with analysis_nonstationary.py")
print("="*80)
