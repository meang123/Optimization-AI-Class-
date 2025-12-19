#!/usr/bin/env python3
"""
Training Progress Collapse Analysis

Parses training logs to extract per-iteration returns and analyzes policy collapse.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.collapse_detector import (
    detect_collapse,
    compute_stability_score,
    analyze_collapse_recovery,
    generate_collapse_summary
)


def parse_training_log(log_path):
    """Parse training log to extract iteration returns."""
    iterations = []
    returns = []

    with open(log_path, 'r') as f:
        for line in f:
            # Match lines like: "Iteration 5/50, Step 99942400, Return=1942.62"
            match = re.search(r'Iteration (\d+)/\d+.*Return=([-\d.]+)', line)
            if match:
                iteration = int(match.group(1))
                return_val = float(match.group(2))
                iterations.append(iteration)
                returns.append(return_val)

    return np.array(iterations), np.array(returns)


def plot_detailed_analysis(model1_data, model2_data, output_path):
    """Generate comprehensive visualization."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle('Training Progress: Policy Collapse Analysis\n'
                 'Model 1 (no L2, minibatch=4) vs Model 2 (L2=1e-4, minibatch=32)',
                 fontsize=16, fontweight='bold')

    iters1 = model1_data['iterations']
    returns1 = model1_data['returns']
    collapses1 = model1_data['collapses']

    iters2 = model2_data['iterations']
    returns2 = model2_data['returns']
    collapses2 = model2_data['collapses']

    # Plot 1: Training curves with collapse markers
    axes[0, 0].plot(iters1, returns1, 'b-o', label='Non-Tested (no L2)',
                   alpha=0.7, markersize=4, linewidth=1.5)
    axes[0, 0].plot(iters2, returns2, 'r-s', label='L2-Reg (1e-4)',
                   alpha=0.7, markersize=4, linewidth=1.5)

    # Mark collapse events
    for event in collapses1:
        axes[0, 0].axvline(x=event['step'], color='blue', alpha=0.3,
                          linestyle='--', linewidth=2, label='Model 1 Collapse' if event == collapses1[0] else '')
    for event in collapses2:
        axes[0, 0].axvline(x=event['step'], color='red', alpha=0.3,
                          linestyle='--', linewidth=2, label='Model 2 Collapse' if event == collapses2[0] else '')

    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    axes[0, 0].set_xlabel('Training Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Episode Return', fontsize=12)
    axes[0, 0].set_title('Training Progress (raw)', fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Smoothed curves
    window = min(5, len(returns1) // 3)
    if window > 1:
        import pandas as pd
        smooth1 = pd.Series(returns1).rolling(window=window, center=True).mean()
        smooth2 = pd.Series(returns2).rolling(window=window, center=True).mean()

        axes[0, 1].plot(iters1, smooth1, 'b-', label='Non-Tested (no L2)', linewidth=2.5)
        axes[0, 1].plot(iters2, smooth2, 'r-', label='L2-Reg (1e-4)', linewidth=2.5)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[0, 1].set_xlabel('Training Iteration', fontsize=12)
        axes[0, 1].set_ylabel('Episode Return (smoothed)', fontsize=12)
        axes[0, 1].set_title(f'Smoothed Training Progress (window={window})',
                            fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc='best', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Not enough data for smoothing',
                       ha='center', va='center', fontsize=12)

    # Plot 3: Performance comparison - first 20 iterations (early phase)
    early_iters = min(20, len(iters1))
    axes[1, 0].plot(iters1[:early_iters], returns1[:early_iters], 'b-o',
                   label='Non-Tested (no L2)', markersize=6, linewidth=2)
    axes[1, 0].plot(iters2[:early_iters], returns2[:early_iters], 'r-s',
                   label='L2-Reg (1e-4)', markersize=6, linewidth=2)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    axes[1, 0].set_xlabel('Training Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Episode Return', fontsize=12)
    axes[1, 0].set_title('Early Training Phase (First 20 Iterations)',
                        fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc='best', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Collapse magnitude comparison
    if collapses1 or collapses2:
        collapse_data = []
        colors = []
        labels = []

        for i, event in enumerate(collapses1):
            collapse_data.append(event['drop_percentage'])
            colors.append('blue')
            labels.append(f"M1-{event['step']}")

        for i, event in enumerate(collapses2):
            collapse_data.append(event['drop_percentage'])
            colors.append('red')
            labels.append(f"M2-{event['step']}")

        if collapse_data:
            x_pos = np.arange(len(collapse_data))
            axes[1, 1].bar(x_pos, collapse_data, color=colors, alpha=0.7)
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            axes[1, 1].set_ylabel('Performance Drop (%)', fontsize=12)
            axes[1, 1].set_title('Collapse Events: Magnitude', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Collapse Events', ha='center', va='center', fontsize=12)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Collapse Events Detected',
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Collapse Events', fontsize=14, fontweight='bold')

    # Plot 5: Distribution comparison
    axes[2, 0].hist(returns1, bins=20, alpha=0.5, label='Non-Tested (no L2)',
                   color='blue', edgecolor='black')
    axes[2, 0].hist(returns2, bins=20, alpha=0.5, label='L2-Reg (1e-4)',
                   color='red', edgecolor='black')
    axes[2, 0].axvline(x=np.mean(returns1), color='blue', linestyle='--',
                      linewidth=2, label=f'Mean M1: {np.mean(returns1):.1f}')
    axes[2, 0].axvline(x=np.mean(returns2), color='red', linestyle='--',
                      linewidth=2, label=f'Mean M2: {np.mean(returns2):.1f}')
    axes[2, 0].set_xlabel('Episode Return', fontsize=12)
    axes[2, 0].set_ylabel('Frequency', fontsize=12)
    axes[2, 0].set_title('Return Distribution', fontsize=14, fontweight='bold')
    axes[2, 0].legend(loc='best', fontsize=9)
    axes[2, 0].grid(True, alpha=0.3, axis='y')

    # Plot 6: Summary statistics
    axes[2, 1].axis('off')

    stability1 = model1_data['stability']
    stability2 = model2_data['stability']
    summary1 = model1_data['summary']
    summary2 = model2_data['summary']

    # Calculate additional metrics
    positive_ratio1 = np.sum(returns1 > 0) / len(returns1) * 100
    positive_ratio2 = np.sum(returns2 > 0) / len(returns2) * 100

    # Find peak performance
    peak1 = np.max(returns1)
    peak1_iter = iters1[np.argmax(returns1)]
    peak2 = np.max(returns2)
    peak2_iter = iters2[np.argmax(returns2)]

    summary_text = f"""
    COMPREHENSIVE ANALYSIS SUMMARY
    {'='*55}

    MODEL 1: Non-Tested (no L2, minibatch=4)
    {'-'*55}
    Total Iterations:       {len(returns1)}
    Collapse Events:        {summary1['total_collapses']}
    Mean Drop (collapse):   {summary1['mean_drop']:.1f}%

    Performance:
      Mean Return:          {np.mean(returns1):.2f} ± {np.std(returns1):.2f}
      Median Return:        {np.median(returns1):.2f}
      Min/Max:              {np.min(returns1):.2f} / {np.max(returns1):.2f}
      Peak at Iter:         {peak1:.2f} (iter {peak1_iter})
      Positive Returns:     {positive_ratio1:.1f}%

    Stability Score:        {stability1:.4f}

    {'='*55}

    MODEL 2: L2-Reg (1e-4, minibatch=32)
    {'-'*55}
    Total Iterations:       {len(returns2)}
    Collapse Events:        {summary2['total_collapses']}
    Mean Drop (collapse):   {summary2['mean_drop']:.1f}%

    Performance:
      Mean Return:          {np.mean(returns2):.2f} ± {np.std(returns2):.2f}
      Median Return:        {np.median(returns2):.2f}
      Min/Max:              {np.min(returns2):.2f} / {np.max(returns2):.2f}
      Peak at Iter:         {peak2:.2f} (iter {peak2_iter})
      Positive Returns:     {positive_ratio2:.1f}%

    Stability Score:        {stability2:.4f}

    {'='*55}
    WINNER: {'Model 2 (L2-Reg)' if np.mean(returns2) > np.mean(returns1) else 'Model 1 (Non-Tested)'}
    Improvement: {abs(np.mean(returns2) - np.mean(returns1)):.2f}
    """

    axes[2, 1].text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("TRAINING PROGRESS COLLAPSE ANALYSIS")
    print("="*70)

    # Paths
    log1_path = Path("logs/ppo_ant_non_tested/1/log.txt")
    log2_path = Path("logs/ppo_ant_nonstationary_test_learned_l2reg/1/log.txt")

    print(f"\nModel 1: {log1_path}")
    print(f"Model 2: {log2_path}")

    # Parse logs
    print("\n📄 Parsing training logs...")
    iters1, returns1 = parse_training_log(log1_path)
    iters2, returns2 = parse_training_log(log2_path)

    print(f"\nModel 1: {len(returns1)} iterations")
    print(f"Model 2: {len(returns2)} iterations")

    # Show training trajectories
    print("\n" + "="*70)
    print("TRAINING TRAJECTORIES")
    print("="*70)

    print("\nModel 1 (Non-Tested, no L2):")
    print(f"  Iter 0-5:   {returns1[0]:.2f} → {returns1[5]:.2f} (early learning)")
    if len(returns1) > 10:
        print(f"  Iter 6-10:  {returns1[6]:.2f} → {returns1[10]:.2f}")
    if len(returns1) > 20:
        print(f"  Iter 11-20: {returns1[11]:.2f} → {returns1[20]:.2f}")
    if len(returns1) > 50:
        print(f"  Iter 21-50: {returns1[21]:.2f} → {returns1[50]:.2f}")
    else:
        print(f"  Final:      {returns1[-1]:.2f}")

    print("\nModel 2 (L2-Reg 1e-4):")
    print(f"  Iter 0-5:   {returns2[0]:.2f} → {returns2[5]:.2f} (early learning)")
    if len(returns2) > 10:
        print(f"  Iter 6-10:  {returns2[6]:.2f} → {returns2[10]:.2f}")
    if len(returns2) > 20:
        print(f"  Iter 11-20: {returns2[11]:.2f} → {returns2[20]:.2f}")
    if len(returns2) > 50:
        print(f"  Iter 21-50: {returns2[21]:.2f} → {returns2[50]:.2f}")
    else:
        print(f"  Final:      {returns2[-1]:.2f}")

    # Detect collapses
    print("\n" + "="*70)
    print("COLLAPSE DETECTION")
    print("="*70)

    # Use smaller window for limited data
    window_size = min(10, len(returns1) // 3)
    threshold = 0.5  # 50% drop
    min_baseline = min(5, len(returns1) // 5)

    print(f"\nParameters:")
    print(f"  Window size: {window_size}")
    print(f"  Threshold: {threshold*100}% performance drop")
    print(f"  Min baseline: {min_baseline} iterations")

    collapses1 = detect_collapse(returns1, window_size=window_size,
                                 threshold=threshold, min_baseline_length=min_baseline)
    collapses2 = detect_collapse(returns2, window_size=window_size,
                                 threshold=threshold, min_baseline_length=min_baseline)

    print(f"\n🔴 Model 1: {len(collapses1)} collapse events detected")
    print(f"🔵 Model 2: {len(collapses2)} collapse events detected")

    # Detailed collapse info
    if collapses1:
        print("\nModel 1 Collapse Events:")
        for i, event in enumerate(collapses1, 1):
            print(f"  {i}. Iteration {event['step']}: {event['drop_percentage']:.1f}% drop")
            print(f"     Baseline: {event['baseline']:.2f} → Collapsed: {event['collapsed_value']:.2f}")

    if collapses2:
        print("\nModel 2 Collapse Events:")
        for i, event in enumerate(collapses2, 1):
            print(f"  {i}. Iteration {event['step']}: {event['drop_percentage']:.1f}% drop")
            print(f"     Baseline: {event['baseline']:.2f} → Collapsed: {event['collapsed_value']:.2f}")

    # Stability analysis
    print("\n" + "="*70)
    print("STABILITY ANALYSIS")
    print("="*70)

    stability1 = compute_stability_score(returns1, window_size=window_size)
    stability2 = compute_stability_score(returns2, window_size=window_size)

    print(f"\nStability Scores (lower = more stable):")
    print(f"  Model 1: {stability1:.4f}")
    print(f"  Model 2: {stability2:.4f}")

    if stability1 < stability2:
        print(f"  → Model 1 is MORE stable")
    else:
        print(f"  → Model 2 is MORE stable")

    # Recovery analysis
    if collapses1:
        print("\n" + "="*70)
        print("RECOVERY ANALYSIS - Model 1")
        print("="*70)

        recovery1 = analyze_collapse_recovery(returns1, collapses1,
                                              recovery_window=20, recovery_threshold=0.8)
        recovered = sum(1 for r in recovery1 if r['recovered'])
        print(f"\nRecovered: {recovered}/{len(collapses1)} collapses")

        for rec in recovery1:
            status = "✓ RECOVERED" if rec['recovered'] else "✗ NOT RECOVERED"
            print(f"  Iter {rec['collapse_step']}: {status}")
            if rec['recovered']:
                print(f"    Recovery time: {rec['recovery_time']} iterations")
            print(f"    Max recovery: {rec['recovery_percentage']:.1f}% of baseline")

    if collapses2:
        print("\n" + "="*70)
        print("RECOVERY ANALYSIS - Model 2")
        print("="*70)

        recovery2 = analyze_collapse_recovery(returns2, collapses2,
                                              recovery_window=20, recovery_threshold=0.8)
        recovered = sum(1 for r in recovery2 if r['recovered'])
        print(f"\nRecovered: {recovered}/{len(collapses2)} collapses")

        for rec in recovery2:
            status = "✓ RECOVERED" if rec['recovered'] else "✗ NOT RECOVERED"
            print(f"  Iter {rec['collapse_step']}: {status}")
            if rec['recovered']:
                print(f"    Recovery time: {rec['recovery_time']} iterations")
            print(f"    Max recovery: {rec['recovery_percentage']:.1f}% of baseline")

    # Performance summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)

    print(f"\nModel 1 (Non-Tested, no L2, minibatch=4):")
    print(f"  Mean:   {np.mean(returns1):.2f} ± {np.std(returns1):.2f}")
    print(f"  Median: {np.median(returns1):.2f}")
    print(f"  Range:  [{np.min(returns1):.2f}, {np.max(returns1):.2f}]")
    print(f"  Positive returns: {np.sum(returns1 > 0)}/{len(returns1)} "
          f"({np.sum(returns1 > 0)/len(returns1)*100:.1f}%)")

    print(f"\nModel 2 (L2-Reg 1e-4, minibatch=32):")
    print(f"  Mean:   {np.mean(returns2):.2f} ± {np.std(returns2):.2f}")
    print(f"  Median: {np.median(returns2):.2f}")
    print(f"  Range:  [{np.min(returns2):.2f}, {np.max(returns2):.2f}]")
    print(f"  Positive returns: {np.sum(returns2 > 0)}/{len(returns2)} "
          f"({np.sum(returns2 > 0)/len(returns2)*100:.1f}%)")

    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    criteria_scores = {'Model 1': 0, 'Model 2': 0}

    print("\nEvaluation Criteria:")

    # 1. Fewer collapses
    if len(collapses1) < len(collapses2):
        criteria_scores['Model 1'] += 1
        print(f"  ✓ Model 1 has FEWER collapses ({len(collapses1)} vs {len(collapses2)})")
    elif len(collapses2) < len(collapses1):
        criteria_scores['Model 2'] += 1
        print(f"  ✓ Model 2 has FEWER collapses ({len(collapses2)} vs {len(collapses1)})")
    else:
        print(f"  = Both have EQUAL collapses ({len(collapses1)})")

    # 2. Better stability
    if stability1 < stability2:
        criteria_scores['Model 1'] += 1
        print(f"  ✓ Model 1 is MORE STABLE ({stability1:.4f} vs {stability2:.4f})")
    else:
        criteria_scores['Model 2'] += 1
        print(f"  ✓ Model 2 is MORE STABLE ({stability2:.4f} vs {stability1:.4f})")

    # 3. Higher mean performance
    if np.mean(returns1) > np.mean(returns2):
        criteria_scores['Model 1'] += 1
        print(f"  ✓ Model 1 has HIGHER mean return ({np.mean(returns1):.2f} vs {np.mean(returns2):.2f})")
    else:
        criteria_scores['Model 2'] += 1
        print(f"  ✓ Model 2 has HIGHER mean return ({np.mean(returns2):.2f} vs {np.mean(returns1):.2f})")

    # 4. Better peak performance
    if np.max(returns1) > np.max(returns2):
        criteria_scores['Model 1'] += 1
        print(f"  ✓ Model 1 achieved HIGHER peak ({np.max(returns1):.2f} vs {np.max(returns2):.2f})")
    else:
        criteria_scores['Model 2'] += 1
        print(f"  ✓ Model 2 achieved HIGHER peak ({np.max(returns2):.2f} vs {np.max(returns1):.2f})")

    # 5. More positive returns
    pos_ratio1 = np.sum(returns1 > 0) / len(returns1)
    pos_ratio2 = np.sum(returns2 > 0) / len(returns2)
    if pos_ratio1 > pos_ratio2:
        criteria_scores['Model 1'] += 1
        print(f"  ✓ Model 1 has MORE positive returns ({pos_ratio1*100:.1f}% vs {pos_ratio2*100:.1f}%)")
    else:
        criteria_scores['Model 2'] += 1
        print(f"  ✓ Model 2 has MORE positive returns ({pos_ratio2*100:.1f}% vs {pos_ratio1*100:.1f}%)")

    print(f"\nFinal Score: Model 1 = {criteria_scores['Model 1']}, Model 2 = {criteria_scores['Model 2']}")

    if criteria_scores['Model 1'] > criteria_scores['Model 2']:
        winner = "Model 1 (Non-Tested, no L2)"
        print(f"\n🏆 WINNER: {winner}")
    elif criteria_scores['Model 2'] > criteria_scores['Model 1']:
        winner = "Model 2 (L2-Reg)"
        print(f"\n🏆 WINNER: {winner}")
    else:
        print("\n🤝 TIE: Both models perform similarly")

    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    # Check for catastrophic collapse
    early_peak1 = np.max(returns1[:10])
    late_mean1 = np.mean(returns1[10:]) if len(returns1) > 10 else returns1[-1]
    early_peak2 = np.max(returns2[:10])
    late_mean2 = np.mean(returns2[10:]) if len(returns2) > 10 else returns2[-1]

    if early_peak1 > 1000 and late_mean1 < 0:
        print("\n⚠️  Model 1 experienced CATASTROPHIC COLLAPSE!")
        print(f"    Early peak: {early_peak1:.2f} → Late mean: {late_mean1:.2f}")
        print(f"    Performance degradation: {((early_peak1 - late_mean1) / early_peak1 * 100):.1f}%")

    if early_peak2 > 1000 and late_mean2 < 0:
        print("\n⚠️  Model 2 experienced CATASTROPHIC COLLAPSE!")
        print(f"    Early peak: {early_peak2:.2f} → Late mean: {late_mean2:.2f}")
        print(f"    Performance degradation: {((early_peak2 - late_mean2) / early_peak2 * 100):.1f}%")

    # L2 regularization effect
    print("\n📝 L2 Regularization Effect:")
    if np.mean(returns2) > np.mean(returns1):
        improvement = np.mean(returns2) - np.mean(returns1)
        print(f"    ✓ L2 regularization IMPROVED performance by {improvement:.2f}")
        print(f"    ✓ This represents a {abs(improvement / np.mean(returns1) * 100):.1f}% improvement")
    else:
        degradation = np.mean(returns1) - np.mean(returns2)
        print(f"    ✗ L2 regularization DEGRADED performance by {degradation:.2f}")

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Prepare data
    summary1 = generate_collapse_summary(collapses1)
    summary2 = generate_collapse_summary(collapses2)

    model1_data = {
        'iterations': iters1,
        'returns': returns1,
        'collapses': collapses1,
        'stability': stability1,
        'summary': summary1
    }

    model2_data = {
        'iterations': iters2,
        'returns': returns2,
        'collapses': collapses2,
        'stability': stability2,
        'summary': summary2
    }

    # Generate visualization
    plot_detailed_analysis(model1_data, model2_data,
                          Path("training_collapse_analysis.png"))

    # Save CSV report
    import pandas as pd
    report_data = {
        'Metric': [
            'Total Iterations',
            'Collapse Events',
            'Mean Return',
            'Std Return',
            'Median Return',
            'Min Return',
            'Max Return',
            'Stability Score',
            'Positive Returns (%)',
            'Mean Collapse Drop (%)',
            'Max Collapse Drop (%)'
        ],
        'Model 1 (no L2)': [
            len(returns1),
            len(collapses1),
            f"{np.mean(returns1):.2f}",
            f"{np.std(returns1):.2f}",
            f"{np.median(returns1):.2f}",
            f"{np.min(returns1):.2f}",
            f"{np.max(returns1):.2f}",
            f"{stability1:.4f}",
            f"{pos_ratio1*100:.1f}",
            f"{summary1['mean_drop']:.2f}",
            f"{summary1['max_drop']:.2f}"
        ],
        'Model 2 (L2=1e-4)': [
            len(returns2),
            len(collapses2),
            f"{np.mean(returns2):.2f}",
            f"{np.std(returns2):.2f}",
            f"{np.median(returns2):.2f}",
            f"{np.min(returns2):.2f}",
            f"{np.max(returns2):.2f}",
            f"{stability2:.4f}",
            f"{pos_ratio2*100:.1f}",
            f"{summary2['mean_drop']:.2f}",
            f"{summary2['max_drop']:.2f}"
        ]
    }

    df = pd.DataFrame(report_data)
    csv_path = Path("training_collapse_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"📄 CSV report saved to: {csv_path}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
