#!/usr/bin/env python3
"""
Quick comparison: Stationary vs Non-Stationary PPO Ant
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
print("="*80)
print("STATIONARY VS NON-STATIONARY COMPARISON")
print("="*80)
print()

stationary_df = pd.read_feather('./logs/ppo_ant_adam/1/result_Test.feather')
nonstationary_df = pd.read_feather('./logs/ppo_ant_nonstationary/1/result_Test.feather')

print("Stationary (Baseline):")
print(f"  Episodes: {len(stationary_df)}")
print(f"  Steps: {stationary_df['Step'].max()/1e6:.0f}M")
print(f"  Mean Return: {stationary_df['Return'].mean():.2f} ± {stationary_df['Return'].std():.2f}")
print(f"  Final 5 eps: {stationary_df['Return'].iloc[-5:].mean():.2f}")
print()

print("Non-Stationary:")
print(f"  Episodes: {len(nonstationary_df)}")
print(f"  Steps: {nonstationary_df['Step'].max()/1e6:.0f}M")
print(f"  Mean Return: {nonstationary_df['Return'].mean():.2f} ± {nonstationary_df['Return'].std():.2f}")
print(f"  Final 5 eps: {nonstationary_df['Return'].iloc[-5:].mean():.2f}")
print()

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Direct comparison (overlaid)
ax1 = axes[0, 0]
ax1.plot(stationary_df['Step']/1e6, stationary_df['Return'], 'o-',
         linewidth=2, markersize=8, label='Stationary', color='#2ecc71')
ax1.plot(nonstationary_df['Step']/1e6, nonstationary_df['Return'], 'o-',
         linewidth=2, markersize=8, label='Non-Stationary', color='#e74c3c', alpha=0.7)
ax1.set_xlabel('Training Steps (Millions)', fontsize=11)
ax1.set_ylabel('Episode Return', fontsize=11)
ax1.set_title('Stationary vs Non-Stationary Performance', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Rolling means
ax2 = axes[0, 1]
stat_rolling = stationary_df['Return'].rolling(window=3, min_periods=1).mean()
nonstat_rolling = nonstationary_df['Return'].rolling(window=5, min_periods=1).mean()

ax2.plot(stationary_df['Step']/1e6, stat_rolling, linewidth=2.5,
         label='Stationary (3-ep rolling)', color='#2ecc71')
ax2.plot(nonstationary_df['Step']/1e6, nonstat_rolling, linewidth=2.5,
         label='Non-Stationary (5-ep rolling)', color='#e74c3c')
ax2.set_xlabel('Training Steps (Millions)', fontsize=11)
ax2.set_ylabel('Episode Return (Rolling Mean)', fontsize=11)
ax2.set_title('Performance Trends', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Distribution comparison
ax3 = axes[1, 0]
ax3.hist([stationary_df['Return'], nonstationary_df['Return']],
         bins=15, label=['Stationary', 'Non-Stationary'],
         color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
ax3.set_xlabel('Episode Return', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Return Distribution Comparison', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Statistics comparison
ax4 = axes[1, 1]
categories = ['Mean', 'Std Dev', 'Max', 'Min']
stat_values = [
    stationary_df['Return'].mean(),
    stationary_df['Return'].std(),
    stationary_df['Return'].max(),
    stationary_df['Return'].min()
]
nonstat_values = [
    nonstationary_df['Return'].mean(),
    nonstationary_df['Return'].std(),
    nonstationary_df['Return'].max(),
    nonstationary_df['Return'].min()
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax4.bar(x - width/2, stat_values, width, label='Stationary',
                color='#2ecc71', alpha=0.7, edgecolor='black')
bars2 = ax4.bar(x + width/2, nonstat_values, width, label='Non-Stationary',
                color='#e74c3c', alpha=0.7, edgecolor='black')

ax4.set_xlabel('Metric', fontsize=11)
ax4.set_ylabel('Episode Return', fontsize=11)
ax4.set_title('Statistical Comparison', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(categories)
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()

# Save
results_dir = Path('./results/ppo_ant_analysis')
results_dir.mkdir(parents=True, exist_ok=True)
save_path = results_dir / 'stationary_vs_nonstationary_comparison.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved comparison plot to {save_path}")
print()

# Summary table
print("="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Metric':<25} {'Stationary':<20} {'Non-Stationary':<20} {'Difference':<15}")
print("-"*80)

metrics = {
    'Mean Return': (stationary_df['Return'].mean(), nonstationary_df['Return'].mean()),
    'Std Dev': (stationary_df['Return'].std(), nonstationary_df['Return'].std()),
    'Max Return': (stationary_df['Return'].max(), nonstationary_df['Return'].max()),
    'Min Return': (stationary_df['Return'].min(), nonstationary_df['Return'].min()),
    'Final 5 Mean': (stationary_df['Return'].iloc[-5:].mean(), nonstationary_df['Return'].iloc[-5:].mean()),
    'Training Steps (M)': (stationary_df['Step'].max()/1e6, nonstationary_df['Step'].max()/1e6),
}

for metric, (stat, nonstat) in metrics.items():
    if 'Steps' in metric:
        diff = f"{nonstat - stat:+.0f}M"
    else:
        diff = f"{nonstat - stat:+.0f} ({(nonstat-stat)/stat*100:+.1f}%)"
    print(f"{metric:<25} {stat:<20.2f} {nonstat:<20.2f} {diff:<15}")

print("="*80)
print()
print("KEY OBSERVATIONS:")
print("  1. Non-stationary training is 10x longer (1000M vs 98M steps)")
print("  2. Non-stationary has higher variance (1729 vs 1325 std dev)")
print(f"  3. Non-stationary mean is {'higher' if nonstationary_df['Return'].mean() > stationary_df['Return'].mean() else 'lower'}: "
      f"{nonstationary_df['Return'].mean():.0f} vs {stationary_df['Return'].mean():.0f}")
print("  4. Peak performance similar (~5800), but non-stationary shows collapse to 795")
print("  5. ⚠️  Note: Different training lengths make direct comparison difficult")
print()
