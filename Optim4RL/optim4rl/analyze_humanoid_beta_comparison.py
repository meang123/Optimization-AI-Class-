#!/usr/bin/env python3
"""
Analyze and compare ppo_humanoid_nonstationary vs ppo_humanoid_nonstationary_adam
to check if equal beta values in Adam reduce policy collapse in Humanoid environment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Paths
base_path = Path('/home/maeng/바탕화면/optimization AI/Optim4RL/optim4rl/logs')
exp1_path = base_path / 'ppo_humanoid_nonstationary' / '1' / 'result_Test.feather'
exp2_path = base_path / 'ppo_humanoid_nonstationary_adam' / '1' / 'result_Test.feather'

print("="*80)
print("HUMANOID ADAM BETA COMPARISON ANALYSIS")
print("="*80)
print()

# Check if files exist
if not exp1_path.exists():
    print(f"❌ Error: {exp1_path} not found!")
    sys.exit(1)

if not exp2_path.exists():
    print(f"❌ Error: {exp2_path} not found!")
    sys.exit(1)

# Load data
print("📊 Loading experiment data...")
try:
    df1 = pd.read_feather(exp1_path)
    print(f"✅ ppo_humanoid_nonstationary (default Adam): {len(df1)} iterations")
    print(f"   Columns: {list(df1.columns)}")
    print(f"   First few rows:")
    print(df1.head())
    print()
except Exception as e:
    print(f"❌ Error loading exp1: {e}")
    sys.exit(1)

try:
    df2 = pd.read_feather(exp2_path)
    print(f"✅ ppo_humanoid_nonstationary_adam (b1=b2=0.997): {len(df2)} iterations")
    print(f"   Columns: {list(df2.columns)}")
    print(f"   First few rows:")
    print(df2.head())
    print()
except Exception as e:
    print(f"❌ Error loading exp2: {e}")
    sys.exit(1)

# Check if exp2 has enough data
if len(df2) < 2:
    print("="*80)
    print("⚠️  WARNING: ppo_humanoid_nonstationary_adam INCOMPLETE")
    print("="*80)
    print()
    print(f"Only {len(df2)} iteration(s) completed.")
    print("The experiment appears to have stopped prematurely.")
    print()
    print("Possible reasons:")
    print("  1. Training was interrupted")
    print("  2. Out of memory error")
    print("  3. Configuration error")
    print()
    print("To complete the comparison, please re-run the experiment:")
    print("  python main.py --config_file ./configs/ppo_humanoid_nonstationary_adam.json --config_idx 1")
    print()

    # Still try to show what we have
    print("Showing available data from ppo_humanoid_nonstationary (default Adam):")
    print()

# Extract metrics
print("="*80)
print("TRAINING PERFORMANCE COMPARISON")
print("="*80)
print()

metrics = ['Return', 'Episode Length'] if 'Return' in df1.columns else df1.columns.tolist()
for metric in metrics:
    if metric in df1.columns:
        print(f"📈 {metric}:")
        print(f"  Default Adam (b1≠b2) - {len(df1)} iterations:")
        print(f"    Mean: {df1[metric].mean():.2f}")
        print(f"    Std:  {df1[metric].std():.2f}")
        print(f"    Max:  {df1[metric].max():.2f}")
        print(f"    Min:  {df1[metric].min():.2f}")
        print()

        if metric in df2.columns and len(df2) > 1:
            print(f"  Equal Beta Adam (b1=b2=0.997) - {len(df2)} iterations:")
            print(f"    Mean: {df2[metric].mean():.2f}")
            print(f"    Std:  {df2[metric].std():.2f}")
            print(f"    Max:  {df2[metric].max():.2f}")
            print(f"    Min:  {df2[metric].min():.2f}")
            print()

# Policy collapse analysis for exp1
print("="*80)
print("POLICY COLLAPSE ANALYSIS - DEFAULT ADAM")
print("="*80)
print()

def detect_collapse(returns, threshold_percentile=50):
    """Detect policy collapse based on performance drop."""
    if len(returns) < 3:
        return [], returns.max() if len(returns) > 0 else 0, 0

    max_return = returns.max()
    threshold = np.percentile(returns, threshold_percentile)

    # Find where performance drops significantly
    collapse_points = []
    for i in range(1, len(returns)):
        if returns.iloc[i] < threshold and returns.iloc[i-1] > threshold:
            collapse_points.append(i)

    return collapse_points, max_return, threshold

if 'Return' in df1.columns:
    collapse1, max1, thresh1 = detect_collapse(df1['Return'])
    print("🔍 Default Adam (b1=0.9, b2=0.999) - Humanoid:")
    print(f"  Peak performance: {max1:.2f}")
    print(f"  Collapse threshold (50th percentile): {thresh1:.2f}")
    print(f"  Performance drop: {max1 - df1['Return'].min():.2f}")
    print(f"  Coefficient of variation: {(df1['Return'].std() / df1['Return'].mean()):.3f}")

    # Specific collapse analysis
    returns = df1['Return'].values
    print(f"\n  Performance trajectory:")
    for i in range(min(len(returns), 21)):
        marker = ""
        if i > 0 and returns[i] < returns[i-1] * 0.5:  # 50% drop
            marker = " ⚠️ MAJOR DROP"
        elif i > 0 and returns[i] < returns[i-1] * 0.8:  # 20% drop
            marker = " ⚠️ DROP"
        print(f"    Iter {i:2d}: {returns[i]:8.2f}{marker}")

    if collapse1:
        print(f"\n  ⚠️  Collapse detected at iterations: {collapse1}")
    else:
        print(f"\n  ✅ No major collapse detected")
    print()

    # Check for catastrophic collapse (performance drops below initial)
    initial_return = df1['Return'].iloc[0]
    catastrophic_iters = df1[df1['Return'] < initial_return].index.tolist()
    if catastrophic_iters:
        print(f"  ❌ CATASTROPHIC COLLAPSE: Performance dropped below initial level!")
        print(f"     Iterations: {catastrophic_iters}")
        print(f"     Initial: {initial_return:.2f}")
        print(f"     Worst after training: {df1['Return'].min():.2f}")
    print()

# Create visualization
print("="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)
print()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Humanoid Adam Beta Comparison: Policy Collapse Analysis', fontsize=16, fontweight='bold')

# Plot 1: Return over iterations
ax1 = axes[0, 0]
iterations1 = range(len(df1))
ax1.plot(iterations1, df1['Return'], 'o-', label='Default Adam (b1≠b2)',
         linewidth=2, markersize=6, alpha=0.7, color='#2E86AB')

if len(df2) > 1 and 'Return' in df2.columns:
    iterations2 = range(len(df2))
    ax1.plot(iterations2, df2['Return'], 's-', label='Equal Beta Adam (b1=b2)',
             linewidth=2, markersize=6, alpha=0.7, color='#A23B72')

if 'Return' in df1.columns:
    ax1.axhline(y=thresh1, color='r', linestyle='--', alpha=0.5, label='Collapse threshold')
    ax1.axhline(y=df1['Return'].iloc[0], color='green', linestyle=':', alpha=0.5, label='Initial performance')

ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Return', fontsize=12)
ax1.set_title('Return over Training Iterations', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Performance drops
ax2 = axes[0, 1]
if 'Return' in df1.columns and len(df1) > 1:
    drops1 = []
    for i in range(1, len(df1)):
        drop_pct = ((df1['Return'].iloc[i] - df1['Return'].iloc[i-1]) / abs(df1['Return'].iloc[i-1])) * 100
        drops1.append(drop_pct)

    ax2.bar(range(1, len(df1)), drops1, alpha=0.7, label='Default Adam', color='#2E86AB')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.axhline(y=-50, color='red', linestyle='--', alpha=0.5, label='Critical drop (-50%)')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Performance Change (%)', fontsize=12)
    ax2.set_title('Iteration-to-Iteration Performance Change', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Distribution
ax3 = axes[1, 0]
if 'Return' in df1.columns:
    ax3.hist(df1['Return'], bins=15, alpha=0.6, label='Default Adam',
             edgecolor='black', color='#2E86AB')
    ax3.axvline(df1['Return'].mean(), color='#2E86AB', linestyle='--',
                linewidth=2, label=f"Mean: {df1['Return'].mean():.0f}")

    if len(df2) > 1 and 'Return' in df2.columns:
        ax3.hist(df2['Return'], bins=15, alpha=0.6, label='Equal Beta Adam',
                 edgecolor='black', color='#A23B72')
        ax3.axvline(df2['Return'].mean(), color='#A23B72', linestyle='--',
                    linewidth=2, label=f"Mean (β): {df2['Return'].mean():.0f}")

    ax3.set_xlabel('Return', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Return Distribution', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Cumulative performance
ax4 = axes[1, 1]
if 'Return' in df1.columns:
    cumulative1 = df1['Return'].cumsum()
    ax4.plot(iterations1, cumulative1, 'o-', label='Default Adam (cumulative)',
             linewidth=2, markersize=5, alpha=0.7, color='#2E86AB')

    if len(df2) > 1 and 'Return' in df2.columns:
        cumulative2 = df2['Return'].cumsum()
        iterations2 = range(len(df2))
        ax4.plot(iterations2, cumulative2, 's-', label='Equal Beta Adam (cumulative)',
                 linewidth=2, markersize=5, alpha=0.7, color='#A23B72')

    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Cumulative Return', fontsize=12)
    ax4.set_title('Cumulative Performance (Higher is Better)', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'humanoid_beta_collapse_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved visualization: {output_path}")
print()

# Summary
print("="*80)
print("SUMMARY - HUMANOID ENVIRONMENT")
print("="*80)
print()

if 'Return' in df1.columns:
    print(f"📊 Default Adam (b1=0.9, b2=0.999):")
    print(f"   Mean Return: {df1['Return'].mean():.2f}")
    print(f"   Std Return:  {df1['Return'].std():.2f}")
    print(f"   Peak:        {df1['Return'].max():.2f}")
    print(f"   Worst:       {df1['Return'].min():.2f}")
    print(f"   Range:       {df1['Return'].max() - df1['Return'].min():.2f}")
    print()

if len(df2) > 1 and 'Return' in df2.columns:
    print(f"📊 Equal Beta Adam (b1=b2=0.997):")
    print(f"   Mean Return: {df2['Return'].mean():.2f}")
    print(f"   Std Return:  {df2['Return'].std():.2f}")
    print(f"   Peak:        {df2['Return'].max():.2f}")
    print(f"   Worst:       {df2['Return'].min():.2f}")
    print(f"   Range:       {df2['Return'].max() - df2['Return'].min():.2f}")
    print()
else:
    print("⚠️  Equal Beta Adam experiment incomplete - cannot compare")
    print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()

if catastrophic_iters:
    print("❌ SEVERE POLICY COLLAPSE DETECTED in Humanoid Default Adam!")
    print(f"   Performance dropped below initial level at {len(catastrophic_iters)} iterations")
    print(f"   This indicates catastrophic forgetting or policy degradation")
    print()
else:
    print("✅ No catastrophic collapse in Humanoid Default Adam")
    print()

if len(df2) < 2:
    print("⚠️  Cannot perform full comparison - Equal Beta Adam experiment incomplete")
    print()
    print("Next steps:")
    print("  1. Re-run: python main.py --config_file ./configs/ppo_humanoid_nonstationary_adam.json --config_idx 1")
    print("  2. Monitor for OOM errors or other issues")
    print("  3. Consider reducing batch_size or num_envs if memory is an issue")
else:
    variance_reduction = (1 - df2['Return'].std() / df1['Return'].std()) * 100
    print(f"📊 Variance reduction with equal beta: {variance_reduction:.1f}%")

print()
print("="*80)
