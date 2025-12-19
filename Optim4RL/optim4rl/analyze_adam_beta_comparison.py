#!/usr/bin/env python3
"""
Analyze and compare ppo_ant_nonstationary vs ppo_ant_nonstationary_adam
to check if equal beta values in Adam reduce policy collapse.

Adam beta comparison:
- ppo_ant_nonstationary: b1=0.9, b2=0.999 (default)
- ppo_ant_nonstationary_adam: b1=0.997, b2=0.997 (equal)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
base_path = Path('/home/maeng/바탕화면/optimization AI/Optim4RL/optim4rl/logs')
exp1_path = base_path / 'ppo_ant_nonstationary' / '1' / 'result_Test.feather'
exp2_path = base_path / 'ppo_ant_nonstationary_adam' / '1' / 'result_Test.feather'

print("="*80)
print("ADAM BETA COMPARISON ANALYSIS")
print("="*80)
print()

# Load data
print("📊 Loading experiment data...")
df1 = pd.read_feather(exp1_path)
df2 = pd.read_feather(exp2_path)

print(f"✅ ppo_ant_nonstationary (default Adam): {len(df1)} iterations")
print(f"✅ ppo_ant_nonstationary_adam (b1=b2=0.997): {len(df2)} iterations")
print()

# Extract metrics
print("="*80)
print("TRAINING PERFORMANCE COMPARISON")
print("="*80)
print()

metrics = ['Return', 'Episode Length']
for metric in metrics:
    if metric in df1.columns and metric in df2.columns:
        print(f"📈 {metric}:")
        print(f"  Default Adam (b1≠b2):")
        print(f"    Mean: {df1[metric].mean():.2f}")
        print(f"    Std:  {df1[metric].std():.2f}")
        print(f"    Max:  {df1[metric].max():.2f}")
        print(f"    Min:  {df1[metric].min():.2f}")
        print()
        print(f"  Equal Beta Adam (b1=b2=0.997):")
        print(f"    Mean: {df2[metric].mean():.2f}")
        print(f"    Std:  {df2[metric].std():.2f}")
        print(f"    Max:  {df2[metric].max():.2f}")
        print(f"    Min:  {df2[metric].min():.2f}")
        print()

# Policy collapse analysis
print("="*80)
print("POLICY COLLAPSE ANALYSIS")
print("="*80)
print()

def detect_collapse(returns, threshold_percentile=50):
    """Detect policy collapse based on performance drop."""
    max_return = returns.max()
    threshold = np.percentile(returns, threshold_percentile)

    # Find where performance drops significantly
    collapse_points = []
    for i in range(1, len(returns)):
        if returns.iloc[i] < threshold and returns.iloc[i-1] > threshold:
            collapse_points.append(i)

    return collapse_points, max_return, threshold

# Analyze default Adam
collapse1, max1, thresh1 = detect_collapse(df1['Return'])
print("🔍 Default Adam (b1=0.9, b2=0.999):")
print(f"  Peak performance: {max1:.2f}")
print(f"  Collapse threshold (50th percentile): {thresh1:.2f}")
print(f"  Performance drop: {max1 - df1['Return'].min():.2f}")
print(f"  Coefficient of variation: {(df1['Return'].std() / df1['Return'].mean()):.3f}")
if collapse1:
    print(f"  ⚠️  Collapse detected at iterations: {collapse1}")
else:
    print(f"  ✅ No major collapse detected")
print()

# Analyze equal beta Adam
collapse2, max2, thresh2 = detect_collapse(df2['Return'])
print("🔍 Equal Beta Adam (b1=0.997, b2=0.997):")
print(f"  Peak performance: {max2:.2f}")
print(f"  Collapse threshold (50th percentile): {thresh2:.2f}")
print(f"  Performance drop: {max2 - df2['Return'].min():.2f}")
print(f"  Coefficient of variation: {(df2['Return'].std() / df2['Return'].mean()):.3f}")
if collapse2:
    print(f"  ⚠️  Collapse detected at iterations: {collapse2}")
else:
    print(f"  ✅ No major collapse detected")
print()

# Stability comparison
print("="*80)
print("STABILITY METRICS")
print("="*80)
print()

# Calculate rolling statistics
window = 5
df1['Rolling Mean'] = df1['Return'].rolling(window=window).mean()
df1['Rolling Std'] = df1['Return'].rolling(window=window).std()
df2['Rolling Mean'] = df2['Return'].rolling(window=window).mean()
df2['Rolling Std'] = df2['Return'].rolling(window=window).std()

print(f"📊 Rolling Statistics (window={window}):")
print(f"  Default Adam:")
print(f"    Avg rolling std: {df1['Rolling Std'].mean():.2f}")
print(f"    Max rolling std: {df1['Rolling Std'].max():.2f}")
print()
print(f"  Equal Beta Adam:")
print(f"    Avg rolling std: {df2['Rolling Std'].mean():.2f}")
print(f"    Max rolling std: {df2['Rolling Std'].max():.2f}")
print()

# Improvement ratio
stability_improvement = (df1['Rolling Std'].mean() - df2['Rolling Std'].mean()) / df1['Rolling Std'].mean() * 100
print(f"🎯 Stability Improvement: {stability_improvement:.1f}%")
if stability_improvement > 0:
    print(f"   ✅ Equal beta Adam is MORE stable")
else:
    print(f"   ⚠️  Equal beta Adam is LESS stable")
print()

# Create comparison visualization
print("="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)
print()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Adam Beta Comparison: Policy Collapse Analysis', fontsize=16, fontweight='bold')

# Plot 1: Return over iterations
ax1 = axes[0, 0]
iterations1 = range(len(df1))
iterations2 = range(len(df2))
ax1.plot(iterations1, df1['Return'], 'o-', label='Default Adam (b1≠b2)', linewidth=2, markersize=6, alpha=0.7)
ax1.plot(iterations2, df2['Return'], 's-', label='Equal Beta Adam (b1=b2)', linewidth=2, markersize=6, alpha=0.7)
ax1.axhline(y=thresh1, color='r', linestyle='--', alpha=0.5, label='Collapse threshold (default)')
ax1.axhline(y=thresh2, color='b', linestyle='--', alpha=0.5, label='Collapse threshold (equal beta)')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Return', fontsize=12)
ax1.set_title('Return over Training Iterations', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Rolling standard deviation
ax2 = axes[0, 1]
ax2.plot(iterations1, df1['Rolling Std'], 'o-', label='Default Adam (b1≠b2)', linewidth=2, markersize=5, alpha=0.7)
ax2.plot(iterations2, df2['Rolling Std'], 's-', label='Equal Beta Adam (b1=b2)', linewidth=2, markersize=5, alpha=0.7)
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Rolling Std (5-iter window)', fontsize=12)
ax2.set_title('Performance Stability (Lower is Better)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distribution comparison
ax3 = axes[1, 0]
ax3.hist(df1['Return'], bins=15, alpha=0.6, label='Default Adam (b1≠b2)', edgecolor='black')
ax3.hist(df2['Return'], bins=15, alpha=0.6, label='Equal Beta Adam (b1=b2)', edgecolor='black')
ax3.axvline(df1['Return'].mean(), color='red', linestyle='--', linewidth=2, label='Mean (default)')
ax3.axvline(df2['Return'].mean(), color='blue', linestyle='--', linewidth=2, label='Mean (equal beta)')
ax3.set_xlabel('Return', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Return Distribution', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Normalized comparison
ax4 = axes[1, 1]
# Normalize to first iteration
df1_norm = (df1['Return'] - df1['Return'].iloc[0]) / df1['Return'].iloc[0] * 100
df2_norm = (df2['Return'] - df2['Return'].iloc[0]) / df2['Return'].iloc[0] * 100
ax4.plot(iterations1, df1_norm, 'o-', label='Default Adam (b1≠b2)', linewidth=2, markersize=6, alpha=0.7)
ax4.plot(iterations2, df2_norm, 's-', label='Equal Beta Adam (b1=b2)', linewidth=2, markersize=6, alpha=0.7)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax4.set_xlabel('Iteration', fontsize=12)
ax4.set_ylabel('Improvement from Iteration 0 (%)', fontsize=12)
ax4.set_title('Normalized Performance Improvement', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'adam_beta_collapse_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved visualization: {output_path}")
print()

# Summary statistics table
print("="*80)
print("SUMMARY TABLE")
print("="*80)
print()

summary_data = {
    'Metric': [
        'Mean Return',
        'Std Return',
        'Peak Return',
        'Lowest Return',
        'Performance Drop',
        'Coefficient of Variation',
        'Avg Rolling Std',
        'Collapse Detected'
    ],
    'Default Adam (b1≠b2)': [
        f"{df1['Return'].mean():.2f}",
        f"{df1['Return'].std():.2f}",
        f"{df1['Return'].max():.2f}",
        f"{df1['Return'].min():.2f}",
        f"{df1['Return'].max() - df1['Return'].min():.2f}",
        f"{df1['Return'].std() / df1['Return'].mean():.3f}",
        f"{df1['Rolling Std'].mean():.2f}",
        "Yes" if collapse1 else "No"
    ],
    'Equal Beta Adam (b1=b2)': [
        f"{df2['Return'].mean():.2f}",
        f"{df2['Return'].std():.2f}",
        f"{df2['Return'].max():.2f}",
        f"{df2['Return'].min():.2f}",
        f"{df2['Return'].max() - df2['Return'].min():.2f}",
        f"{df2['Return'].std() / df2['Return'].mean():.3f}",
        f"{df2['Rolling Std'].mean():.2f}",
        "Yes" if collapse2 else "No"
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
print()

# Save summary
summary_df.to_csv('adam_beta_comparison_summary.csv', index=False)
print(f"✅ Saved summary: adam_beta_comparison_summary.csv")
print()

# Final conclusion
print("="*80)
print("CONCLUSION")
print("="*80)
print()

if df2['Return'].std() < df1['Return'].std():
    print("✅ CONFIRMED: Equal beta values (b1=b2=0.997) REDUCE performance variance")
else:
    print("❌ NOT CONFIRMED: Equal beta values did not reduce variance")

if df2['Rolling Std'].mean() < df1['Rolling Std'].mean():
    print("✅ CONFIRMED: Equal beta values IMPROVE stability (lower rolling std)")
else:
    print("❌ NOT CONFIRMED: Equal beta values did not improve stability")

if df2['Return'].max() - df2['Return'].min() < df1['Return'].max() - df1['Return'].min():
    print("✅ CONFIRMED: Equal beta values REDUCE performance drop (policy collapse)")
else:
    print("❌ NOT CONFIRMED: Equal beta values did not reduce performance drop")

print()
print("📊 Overall assessment:")
variance_reduction = (1 - df2['Return'].std() / df1['Return'].std()) * 100
print(f"  Variance reduction: {variance_reduction:.1f}%")
stability_gain = (1 - df2['Rolling Std'].mean() / df1['Rolling Std'].mean()) * 100
print(f"  Stability improvement: {stability_gain:.1f}%")
collapse_reduction = (1 - (df2['Return'].max() - df2['Return'].min()) / (df1['Return'].max() - df1['Return'].min())) * 100
print(f"  Collapse reduction: {collapse_reduction:.1f}%")
print()

if variance_reduction > 0 and stability_gain > 0:
    print("🎯 VERDICT: Equal beta Adam (b1=b2=0.997) EFFECTIVELY MITIGATES policy collapse!")
else:
    print("⚠️  VERDICT: Results are mixed or inconclusive")
print()

print("="*80)
