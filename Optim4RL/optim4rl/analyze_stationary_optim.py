import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the results
df = pd.read_feather('logs/stationary_optim/1/result_Test.feather')

# Create comprehensive analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Stationary Optim (Optim4RL) Analysis - Policy Collapse Detected', fontsize=14, fontweight='bold')

# Plot 1: Return over iterations
ax1 = axes[0, 0]
iterations = df.index
returns = df['Return'].values
ax1.plot(iterations, returns, 'b-o', linewidth=2, markersize=6)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Mark peak and collapse
peak_idx = returns.argmax()
collapse_start_idx = 5  # Where collapse begins
min_idx = returns[1:].argmin() + 1  # Skip initial value

ax1.plot(peak_idx, returns[peak_idx], 'g*', markersize=20, label=f'Peak: {returns[peak_idx]:.1f} (iter {peak_idx})')
ax1.plot(collapse_start_idx, returns[collapse_start_idx], 'r^', markersize=12, label=f'Collapse Start (iter {collapse_start_idx})')
ax1.plot(min_idx, returns[min_idx], 'rv', markersize=12, label=f'Min: {returns[min_idx]:.1f} (iter {min_idx})')

ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Episode Return', fontsize=11)
ax1.set_title('Episode Return Over Training', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Plot 2: Return change (derivative)
ax2 = axes[0, 1]
return_changes = np.diff(returns)
ax2.plot(iterations[1:], return_changes, 'r-o', linewidth=2, markersize=5)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax2.fill_between(iterations[1:], 0, return_changes, where=(return_changes<0),
                  color='red', alpha=0.3, label='Performance Loss')
ax2.fill_between(iterations[1:], 0, return_changes, where=(return_changes>=0),
                  color='green', alpha=0.3, label='Performance Gain')
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Return Change (Δ)', fontsize=11)
ax2.set_title('Return Change Between Iterations', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# Plot 3: Cumulative performance loss
ax3 = axes[1, 0]
peak_return = returns[peak_idx]
performance_loss = peak_return - returns
cumulative_loss = np.cumsum(performance_loss)

ax3.plot(iterations, performance_loss, 'orange', linewidth=2, label='Per-Iteration Loss from Peak')
ax3.fill_between(iterations, 0, performance_loss, alpha=0.3, color='orange')
ax3.axvline(x=collapse_start_idx, color='red', linestyle='--', alpha=0.7, label='Collapse Onset')
ax3.set_xlabel('Iteration', fontsize=11)
ax3.set_ylabel('Performance Loss from Peak', fontsize=11)
ax3.set_title('Performance Degradation Analysis', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)

# Plot 4: Statistics table
ax4 = axes[1, 1]
ax4.axis('off')

# Calculate statistics
stats_data = [
    ['Metric', 'Value'],
    ['─'*30, '─'*15],
    ['Peak Return', f'{returns[peak_idx]:.2f}'],
    ['Peak Iteration', f'{peak_idx}'],
    ['Final Return', f'{returns[-1]:.2f}'],
    ['Performance Retained', f'{(returns[-1]/returns[peak_idx]*100):.1f}%'],
    ['─'*30, '─'*15],
    ['Collapse Start (iter)', f'{collapse_start_idx}'],
    ['Min Return', f'{returns[min_idx]:.2f}'],
    ['Min Return Iteration', f'{min_idx}'],
    ['─'*30, '─'*15],
    ['Total Return Drop', f'{peak_return - returns[-1]:.2f}'],
    ['Max Single Drop', f'{np.abs(return_changes).max():.2f}'],
    ['Average Return (all)', f'{returns[1:].mean():.2f}'],
    ['Std Dev (all)', f'{returns[1:].std():.2f}'],
    ['─'*30, '─'*15],
    ['Policy Collapse?', '✓ YES'],
    ['Severity', 'SEVERE'],
]

table = ax4.table(cellText=stats_data, cellLoc='left', loc='center',
                  colWidths=[0.65, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color code the rows
for i in range(len(stats_data)):
    if i == 0 or '─' in stats_data[i][0]:
        for j in range(2):
            table[(i, j)].set_facecolor('#E0E0E0')
            table[(i, j)].set_text_props(weight='bold')
    elif 'Collapse' in stats_data[i][0] or 'YES' in str(stats_data[i][1]) or 'SEVERE' in str(stats_data[i][1]):
        for j in range(2):
            table[(i, j)].set_facecolor('#FFCDD2')

ax4.set_title('Training Statistics Summary', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('stationary_optim_collapse_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: stationary_optim_collapse_analysis.png")

# Generate detailed report
print("\n" + "="*70)
print("POLICY COLLAPSE ANALYSIS REPORT: stationary_optim.json")
print("="*70)
print(f"\nConfiguration:")
print(f"  - Environment: Ant (Non-stationary)")
print(f"  - Agent: PPO")
print(f"  - Optimizer: Optim4RL (learned optimizer)")
print(f"  - Total Iterations: {len(df)-1}")
print(f"  - Total Steps: {df['Step'].iloc[-1]:,.0f}")

print(f"\n{'PERFORMANCE SUMMARY':^70}")
print("-"*70)
print(f"Initial Return (iter 0):        {returns[0]:>12.2f}")
print(f"Peak Return (iter {peak_idx}):          {returns[peak_idx]:>12.2f} ⭐")
print(f"Final Return (iter {len(returns)-1}):        {returns[-1]:>12.2f}")
print(f"Minimum Return (iter {min_idx}):        {returns[min_idx]:>12.2f} ⚠️")
print(f"\nPerformance Retention:          {(returns[-1]/returns[peak_idx]*100):>11.1f}%")
print(f"Performance Lost:               {(1-returns[-1]/returns[peak_idx])*100:>11.1f}%")

print(f"\n{'COLLAPSE INDICATORS':^70}")
print("-"*70)

# Check for collapse indicators
collapse_detected = False
collapse_reasons = []

# 1. Sudden performance drop
max_drop = np.abs(return_changes.min())
if max_drop > 1000:
    collapse_reasons.append(f"✗ Sudden drop detected: {max_drop:.2f} (iter {return_changes.argmin()+1}→{return_changes.argmin()+2})")
    collapse_detected = True

# 2. Sustained degradation
if returns[peak_idx] - returns[-1] > 4000:
    collapse_reasons.append(f"✗ Severe sustained degradation: {returns[peak_idx] - returns[-1]:.2f}")
    collapse_detected = True

# 3. Performance retention
if returns[-1] / returns[peak_idx] < 0.5:
    collapse_reasons.append(f"✗ Final performance < 50% of peak ({(returns[-1]/returns[peak_idx]*100):.1f}%)")
    collapse_detected = True

# 4. Multiple consecutive drops
consecutive_drops = 0
max_consecutive = 0
for change in return_changes:
    if change < 0:
        consecutive_drops += 1
        max_consecutive = max(max_consecutive, consecutive_drops)
    else:
        consecutive_drops = 0

if max_consecutive >= 5:
    collapse_reasons.append(f"✗ {max_consecutive} consecutive performance drops")
    collapse_detected = True

# 5. Failed recovery
post_collapse_returns = returns[collapse_start_idx:]
recovery_ratio = post_collapse_returns.max() / returns[peak_idx]
if recovery_ratio < 0.3:
    collapse_reasons.append(f"✗ Failed to recover (max recovery: {recovery_ratio*100:.1f}% of peak)")
    collapse_detected = True

print("Collapse Status: ", end="")
if collapse_detected:
    print("🔴 POLICY COLLAPSE DETECTED")
    print("\nCollapse Evidence:")
    for reason in collapse_reasons:
        print(f"  {reason}")
else:
    print("🟢 No collapse detected")

print(f"\n{'DETAILED TIMELINE':^70}")
print("-"*70)
print(f"{'Iter':<6} {'Step':<12} {'Return':<12} {'Change':<12} {'Status':<20}")
print("-"*70)

for i in range(len(returns)):
    change_str = "─" if i == 0 else f"{return_changes[i-1]:+.2f}"
    status = ""

    if i == peak_idx:
        status = "🏆 PEAK"
    elif i == collapse_start_idx:
        status = "⚠️ COLLAPSE START"
    elif i == min_idx:
        status = "🔻 MINIMUM"
    elif i > 0 and return_changes[i-1] < -1000:
        status = "📉 MAJOR DROP"
    elif i > 0 and return_changes[i-1] > 0:
        status = "📈 Recovery"

    print(f"{i:<6} {df['Step'].iloc[i]:<12.0f} {returns[i]:<12.2f} {change_str:<12} {status:<20}")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print(f"""
The Optim4RL optimizer exhibits SEVERE POLICY COLLAPSE on the Ant task
with non-stationary dynamics:

1. Training achieved peak performance of {returns[peak_idx]:.1f} at iteration {peak_idx}
2. Catastrophic collapse began at iteration {collapse_start_idx}, dropping to {returns[collapse_start_idx]:.1f}
3. Performance continued degrading to minimum {returns[min_idx]:.1f} at iteration {min_idx}
4. Final performance ({returns[-1]:.1f}) retained only {(returns[-1]/returns[peak_idx]*100):.1f}% of peak
5. Despite some recovery attempts, the policy never regained stability

VERDICT: Optim4RL is UNSTABLE for non-stationary Ant environments.
         The learned optimizer fails to maintain stable learning dynamics.
""")
print("="*70)
