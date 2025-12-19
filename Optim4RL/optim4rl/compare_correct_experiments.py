"""
Correct comparison between:
1. ppo_ant_nonstationary (Adam optimizer on non-stationary Ant)
2. ppo_ant_non_tested (Learned Optim4RL on non-stationary Ant)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
baseline_path = "./logs/ppo_ant_nonstationary/1/result_Test.feather"
learned_path = "./logs/ppo_ant_non_tested/1/result_Test.feather"

baseline_df = pd.read_feather(baseline_path)
learned_df = pd.read_feather(learned_path)

print("="*80)
print("EXPERIMENT COMPARISON: Non-Stationary Ant Environment")
print("="*80)
print(f"\n1. Adam Baseline (ppo_ant_nonstationary):")
print(f"   - Optimizer: Adam")
print(f"   - Config: configs/ppo_ant_nonstationary.json")
print(f"   - Data points: {len(baseline_df)}")

print(f"\n2. Learned Optim4RL (ppo_ant_non_tested):")
print(f"   - Optimizer: Learned Optim4RL")
print(f"   - Config: configs/ppo_ant_non_tested.json")
print(f"   - Data points: {len(learned_df)}")

# Print column names to understand data structure
print(f"\nBaseline columns: {baseline_df.columns.tolist()}")
print(f"Learned columns: {learned_df.columns.tolist()}")

# Print first few rows
print("\nBaseline data (first 5 rows):")
print(baseline_df.head())
print("\nLearned data (first 5 rows):")
print(learned_df.head())

# Check for collapse (negative returns for extended periods)
def detect_collapse(df, threshold=-500, min_consecutive=5):
    """Detect if training collapsed based on consecutive negative returns"""
    if 'Return' in df.columns:
        returns = df['Return'].values
    elif 'episode_returns' in df.columns:
        returns = df['episode_returns'].values
    elif 'episode_return' in df.columns:
        returns = df['episode_return'].values
    else:
        print(f"Warning: No return column found. Available: {df.columns.tolist()}")
        return False

    consecutive_negative = 0
    for ret in returns[-20:]:  # Check last 20 evaluations
        if ret < threshold:
            consecutive_negative += 1
        else:
            consecutive_negative = 0
        if consecutive_negative >= min_consecutive:
            return True
    return False

baseline_collapsed = detect_collapse(baseline_df)
learned_collapsed = detect_collapse(learned_df)

# Calculate statistics
def get_stats(df):
    if 'Return' in df.columns:
        returns = df['Return'].values
    elif 'episode_returns' in df.columns:
        returns = df['episode_returns'].values
    elif 'episode_return' in df.columns:
        returns = df['episode_return'].values
    else:
        return None

    return {
        'mean': np.mean(returns),
        'std': np.std(returns),
        'min': np.min(returns),
        'max': np.max(returns),
        'median': np.median(returns),
        'final_10_mean': np.mean(returns[-10:]),
    }

baseline_stats = get_stats(baseline_df)
learned_stats = get_stats(learned_df)

print("\n" + "="*80)
print("STATISTICS")
print("="*80)
print("\nAdam Baseline:")
if baseline_stats:
    for key, val in baseline_stats.items():
        print(f"  {key}: {val:.2f}")
    print(f"  Collapsed: {baseline_collapsed}")

print("\nLearned Optim4RL:")
if learned_stats:
    for key, val in learned_stats.items():
        print(f"  {key}: {val:.2f}")
    print(f"  Collapsed: {learned_collapsed}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Correct Comparison: Adam vs Learned Optim4RL (Non-Stationary Ant)',
             fontsize=16, fontweight='bold')

# Determine return column name
if 'Return' in baseline_df.columns:
    baseline_return_col = 'Return'
elif 'episode_returns' in baseline_df.columns:
    baseline_return_col = 'episode_returns'
else:
    baseline_return_col = 'episode_return'

if 'Return' in learned_df.columns:
    learned_return_col = 'Return'
elif 'episode_returns' in learned_df.columns:
    learned_return_col = 'episode_returns'
else:
    learned_return_col = 'episode_return'

# Plot 1: Episode returns over time
ax1 = axes[0, 0]
if 'Step' in baseline_df.columns:
    ax1.plot(baseline_df['Step'], baseline_df[baseline_return_col],
             label='Adam Baseline', color='green', linewidth=2, alpha=0.8, marker='o')
    ax1.plot(learned_df['Step'], learned_df[learned_return_col],
             label='Learned Optim4RL', color='blue', linewidth=2, alpha=0.8, marker='s')
    ax1.set_xlabel('Training Steps', fontsize=12)
elif 'steps' in baseline_df.columns:
    ax1.plot(baseline_df['steps'], baseline_df[baseline_return_col],
             label='Adam Baseline', color='green', linewidth=2, alpha=0.8)
    ax1.plot(learned_df['steps'], learned_df[learned_return_col],
             label='Learned Optim4RL', color='blue', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Training Steps', fontsize=12)
else:
    ax1.plot(baseline_df[baseline_return_col],
             label='Adam Baseline', color='green', linewidth=2, alpha=0.8)
    ax1.plot(learned_df[learned_return_col],
             label='Learned Optim4RL', color='blue', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Evaluation Episode', fontsize=12)

ax1.set_ylabel('Episode Return', fontsize=12)
ax1.set_title('Episode Returns Over Training', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)

# Plot 2: Return distribution comparison (box plot)
ax2 = axes[0, 1]
box_data = [learned_df[learned_return_col].values, baseline_df[baseline_return_col].values]
box_colors = ['lightblue', 'lightgreen']
bp = ax2.boxplot(box_data, labels=['Learned Optim4RL', 'Adam Baseline'],
                 patch_artist=True, showfliers=True)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
ax2.set_ylabel('Episode Return', fontsize=12)
ax2.set_title('Return Distribution Comparison', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)

# Plot 3: Return distribution histograms
ax3 = axes[1, 0]
ax3.hist(learned_df[learned_return_col].values, bins=30, alpha=0.6,
         label='Learned Optim4RL', color='blue', edgecolor='black')
ax3.hist(baseline_df[baseline_return_col].values, bins=30, alpha=0.6,
         label='Adam Baseline', color='green', edgecolor='black')
ax3.set_xlabel('Episode Return', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Return Distribution', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')
ax3.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)

# Plot 4: Performance statistics table
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

table_data = []
table_data.append(['Metric', 'Learned Optim4RL', 'Adam Baseline'])
table_data.append(['Mean', f'{learned_stats["mean"]:.2f}', f'{baseline_stats["mean"]:.2f}'])
table_data.append(['Std', f'{learned_stats["std"]:.2f}', f'{baseline_stats["std"]:.2f}'])
table_data.append(['Min', f'{learned_stats["min"]:.2f}', f'{baseline_stats["min"]:.2f}'])
table_data.append(['Max', f'{learned_stats["max"]:.2f}', f'{baseline_stats["max"]:.2f}'])
table_data.append(['Median', f'{learned_stats["median"]:.2f}', f'{baseline_stats["median"]:.2f}'])
table_data.append(['Final 10 Mean', f'{learned_stats["final_10_mean"]:.2f}',
                   f'{baseline_stats["final_10_mean"]:.2f}'])
table_data.append(['Collapse?',
                   '⚠ YES' if learned_collapsed else '✓ NO',
                   '⚠ YES' if baseline_collapsed else '✓ NO'])

# Color code the header
colors = [['lightgray', 'lightblue', 'lightgreen']]
for _ in range(len(table_data) - 1):
    colors.append(['white', 'white', 'white'])

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                  cellColours=colors)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Bold header row
for i in range(3):
    table[(0, i)].set_text_props(weight='bold')

ax4.set_title('Performance Statistics', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('./results/correct_comparison_adam_vs_learned.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to: ./results/correct_comparison_adam_vs_learned.png")

# Save comparison report
report = []
report.append("="*80)
report.append("CORRECT EXPERIMENT COMPARISON REPORT")
report.append("="*80)
report.append(f"\nGenerated: {pd.Timestamp.now()}")
report.append(f"\nExperiment 1: Adam Baseline (ppo_ant_nonstationary)")
report.append(f"  - Config: configs/ppo_ant_nonstationary.json")
report.append(f"  - Optimizer: Adam")
report.append(f"  - Environment: Non-stationary Ant")
report.append(f"\nExperiment 2: Learned Optim4RL (ppo_ant_non_tested)")
report.append(f"  - Config: configs/ppo_ant_non_tested.json")
report.append(f"  - Optimizer: Learned Optim4RL")
report.append(f"  - Environment: Non-stationary Ant")
report.append("\n" + "="*80)
report.append("RESULTS")
report.append("="*80)
report.append(f"\nAdam Baseline:")
for key, val in baseline_stats.items():
    report.append(f"  {key}: {val:.2f}")
report.append(f"  Collapsed: {baseline_collapsed}")
report.append(f"\nLearned Optim4RL:")
for key, val in learned_stats.items():
    report.append(f"  {key}: {val:.2f}")
report.append(f"  Collapsed: {learned_collapsed}")

# Performance comparison
improvement = ((learned_stats['mean'] - baseline_stats['mean']) / abs(baseline_stats['mean'])) * 100
report.append("\n" + "="*80)
report.append("ANALYSIS")
report.append("="*80)
report.append(f"\nPerformance Difference:")
report.append(f"  Learned Optim4RL vs Adam: {improvement:+.2f}%")
if learned_stats['mean'] > baseline_stats['mean']:
    report.append(f"  ✓ Learned optimizer is BETTER by {abs(improvement):.2f}%")
else:
    report.append(f"  ✗ Learned optimizer is WORSE by {abs(improvement):.2f}%")

report.append(f"\nCollapse Detection:")
if not learned_collapsed and not baseline_collapsed:
    report.append("  ✓ Both experiments completed successfully without collapse")
elif learned_collapsed and not baseline_collapsed:
    report.append("  ✗ Learned optimizer collapsed, Adam remained stable")
elif not learned_collapsed and baseline_collapsed:
    report.append("  ⚠ Adam collapsed, but learned optimizer remained stable")
else:
    report.append("  ✗ Both experiments collapsed")

report_text = "\n".join(report)
with open('./results/correct_comparison_report.txt', 'w') as f:
    f.write(report_text)

print(report_text)
print(f"\n✓ Report saved to: ./results/correct_comparison_report.txt")

plt.show()
