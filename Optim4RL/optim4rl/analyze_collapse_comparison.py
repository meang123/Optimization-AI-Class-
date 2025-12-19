#!/usr/bin/env python3
"""
Policy Collapse Analysis: Comparing ppo_ant_non_tested vs ppo_ant_nonstationary_test_learned_l2reg

This script analyzes training results to detect policy collapse using the collapse_detector utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.collapse_detector import (
    detect_collapse,
    compute_stability_score,
    analyze_collapse_recovery,
    compare_collapse_rates,
    generate_collapse_summary
)


def load_training_results(result_path):
    """Load training results from feather file."""
    df = pd.read_feather(result_path)
    return df


def extract_episode_rewards(df):
    """Extract episode rewards from training dataframe."""
    # Assuming the dataframe has a column for episode rewards
    # Common column names: 'Return', 'eval/episode_reward', 'episode_reward', 'reward', etc.

    # First try 'Return' column (common in this codebase)
    if 'Return' in df.columns:
        print(f"Using reward column: Return")
        return df['Return'].values

    # Try to find reward-related columns
    reward_cols = [col for col in df.columns if 'reward' in col.lower() and 'episode' in col.lower()]

    if not reward_cols:
        # Try alternative column names
        reward_cols = [col for col in df.columns if 'reward' in col.lower() or 'return' in col.lower()]

    if not reward_cols:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError("Could not find reward column in dataframe")

    reward_col = reward_cols[0]
    print(f"Using reward column: {reward_col}")

    return df[reward_col].values


def plot_collapse_analysis(model1_data, model2_data, output_path):
    """Generate comprehensive visualization of collapse analysis."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Policy Collapse Analysis: Non-Tested vs L2-Reg', fontsize=16, fontweight='bold')

    model1_rewards = model1_data['rewards']
    model2_rewards = model2_data['rewards']
    model1_collapses = model1_data['collapses']
    model2_collapses = model2_data['collapses']

    # Plot 1: Training curves
    axes[0, 0].plot(model1_rewards, label='Non-Tested (no L2)', alpha=0.7, linewidth=1)
    axes[0, 0].plot(model2_rewards, label='L2-Reg (1e-4)', alpha=0.7, linewidth=1)

    # Mark collapse events
    for event in model1_collapses:
        axes[0, 0].axvline(x=event['step'], color='red', alpha=0.3, linestyle='--', linewidth=1)
    for event in model2_collapses:
        axes[0, 0].axvline(x=event['step'], color='orange', alpha=0.3, linestyle='--', linewidth=1)

    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('Episode Reward')
    axes[0, 0].set_title('Training Curves (vertical lines = collapse events)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Rolling mean comparison
    window = 100
    model1_rolling = pd.Series(model1_rewards).rolling(window=window).mean()
    model2_rolling = pd.Series(model2_rewards).rolling(window=window).mean()

    axes[0, 1].plot(model1_rolling, label='Non-Tested (no L2)', linewidth=2)
    axes[0, 1].plot(model2_rolling, label='L2-Reg (1e-4)', linewidth=2)
    axes[0, 1].set_xlabel('Training Steps')
    axes[0, 1].set_ylabel('Episode Reward (Rolling Mean)')
    axes[0, 1].set_title(f'Smoothed Training Curves (window={window})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Collapse events comparison
    if model1_collapses or model2_collapses:
        collapse_steps_1 = [e['step'] for e in model1_collapses]
        collapse_drops_1 = [e['drop_percentage'] for e in model1_collapses]
        collapse_steps_2 = [e['step'] for e in model2_collapses]
        collapse_drops_2 = [e['drop_percentage'] for e in model2_collapses]

        axes[1, 0].scatter(collapse_steps_1, collapse_drops_1,
                          color='red', s=100, alpha=0.6, label='Non-Tested (no L2)', marker='o')
        axes[1, 0].scatter(collapse_steps_2, collapse_drops_2,
                          color='orange', s=100, alpha=0.6, label='L2-Reg (1e-4)', marker='s')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Performance Drop (%)')
        axes[1, 0].set_title('Collapse Events: Magnitude and Timing')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Collapse Events Detected',
                       ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('Collapse Events')

    # Plot 4: Stability comparison (rolling std)
    window = 100
    model1_rolling_std = pd.Series(model1_rewards).rolling(window=window).std()
    model2_rolling_std = pd.Series(model2_rewards).rolling(window=window).std()

    axes[1, 1].plot(model1_rolling_std, label='Non-Tested (no L2)', alpha=0.7)
    axes[1, 1].plot(model2_rolling_std, label='L2-Reg (1e-4)', alpha=0.7)
    axes[1, 1].set_xlabel('Training Steps')
    axes[1, 1].set_ylabel('Rolling Std Dev')
    axes[1, 1].set_title(f'Performance Stability (window={window})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Distribution comparison
    axes[2, 0].hist(model1_rewards, bins=50, alpha=0.5, label='Non-Tested (no L2)', color='red')
    axes[2, 0].hist(model2_rewards, bins=50, alpha=0.5, label='L2-Reg (1e-4)', color='orange')
    axes[2, 0].set_xlabel('Episode Reward')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].set_title('Reward Distribution')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3, axis='y')

    # Plot 6: Summary statistics
    axes[2, 1].axis('off')

    model1_stability = model1_data['stability']
    model2_stability = model2_data['stability']
    model1_summary = model1_data['summary']
    model2_summary = model2_data['summary']

    summary_text = f"""
    COLLAPSE ANALYSIS SUMMARY
    {'='*50}

    Model 1: Non-Tested (no L2, minibatch=4)
    - Total Collapses: {model1_summary['total_collapses']}
    - Mean Drop: {model1_summary['mean_drop']:.2f}%
    - Max Drop: {model1_summary['max_drop']:.2f}%
    - Stability Score: {model1_stability:.4f}
    - Mean Reward: {np.mean(model1_rewards):.2f}
    - Std Reward: {np.std(model1_rewards):.2f}

    Model 2: L2-Reg (1e-4, minibatch=32)
    - Total Collapses: {model2_summary['total_collapses']}
    - Mean Drop: {model2_summary['mean_drop']:.2f}%
    - Max Drop: {model2_summary['max_drop']:.2f}%
    - Stability Score: {model2_stability:.4f}
    - Mean Reward: {np.mean(model2_rewards):.2f}
    - Std Reward: {np.std(model2_rewards):.2f}

    {'='*50}
    INTERPRETATION
    {'='*50}
    Lower stability score = more stable
    Fewer collapses = better robustness
    Higher mean reward = better performance
    """

    axes[2, 1].text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def main():
    # Paths to training results
    model1_path = Path("logs/ppo_ant_non_tested/1/result_Test.feather")
    model2_path = Path("logs/ppo_ant_nonstationary_test_learned_l2reg/1/result_Test.feather")

    print("="*70)
    print("POLICY COLLAPSE ANALYSIS")
    print("="*70)
    print(f"\nModel 1: {model1_path}")
    print(f"Model 2: {model2_path}")
    print()

    # Load training results
    print("Loading training results...")
    df1 = load_training_results(model1_path)
    df2 = load_training_results(model2_path)

    print(f"\nModel 1 columns: {df1.columns.tolist()}")
    print(f"Model 2 columns: {df2.columns.tolist()}")

    # Extract episode rewards
    print("\nExtracting episode rewards...")
    print("Model 1:")
    model1_rewards = extract_episode_rewards(df1)
    print("Model 2:")
    model2_rewards = extract_episode_rewards(df2)

    print(f"\nModel 1: {len(model1_rewards)} episodes")
    print(f"Model 2: {len(model2_rewards)} episodes")

    # Detect collapses
    print("\n" + "="*70)
    print("DETECTING POLICY COLLAPSES")
    print("="*70)

    # Use different thresholds to detect collapses
    threshold = 0.5  # 50% drop
    window_size = 100

    print(f"\nCollapse detection parameters:")
    print(f"  - Window size: {window_size}")
    print(f"  - Threshold: {threshold*100}% performance drop")

    model1_collapses = detect_collapse(model1_rewards, window_size=window_size, threshold=threshold)
    model2_collapses = detect_collapse(model2_rewards, window_size=window_size, threshold=threshold)

    print(f"\nModel 1 (Non-Tested): {len(model1_collapses)} collapse events detected")
    print(f"Model 2 (L2-Reg): {len(model2_collapses)} collapse events detected")

    # Detailed collapse information
    if model1_collapses:
        print("\nModel 1 Collapse Events:")
        for i, event in enumerate(model1_collapses[:5]):  # Show first 5
            print(f"  {i+1}. Step {event['step']}: {event['drop_percentage']:.1f}% drop "
                  f"(baseline: {event['baseline']:.2f} → {event['collapsed_value']:.2f})")
        if len(model1_collapses) > 5:
            print(f"  ... and {len(model1_collapses) - 5} more")

    if model2_collapses:
        print("\nModel 2 Collapse Events:")
        for i, event in enumerate(model2_collapses[:5]):  # Show first 5
            print(f"  {i+1}. Step {event['step']}: {event['drop_percentage']:.1f}% drop "
                  f"(baseline: {event['baseline']:.2f} → {event['collapsed_value']:.2f})")
        if len(model2_collapses) > 5:
            print(f"  ... and {len(model2_collapses) - 5} more")

    # Compute stability scores
    print("\n" + "="*70)
    print("STABILITY ANALYSIS")
    print("="*70)

    model1_stability = compute_stability_score(model1_rewards, window_size=100)
    model2_stability = compute_stability_score(model2_rewards, window_size=100)

    print(f"\nStability Scores (lower is better):")
    print(f"  Model 1 (Non-Tested): {model1_stability:.4f}")
    print(f"  Model 2 (L2-Reg): {model2_stability:.4f}")

    if model1_stability < model2_stability:
        print(f"  → Model 1 is MORE stable ({(model2_stability/model1_stability - 1)*100:.1f}% better)")
    else:
        print(f"  → Model 2 is MORE stable ({(model1_stability/model2_stability - 1)*100:.1f}% better)")

    # Recovery analysis
    if model1_collapses:
        print("\n" + "="*70)
        print("RECOVERY ANALYSIS - Model 1")
        print("="*70)

        model1_recovery = analyze_collapse_recovery(model1_rewards, model1_collapses)
        recovered_count = sum(1 for r in model1_recovery if r['recovered'])
        print(f"\nRecovered from {recovered_count}/{len(model1_collapses)} collapses")

        for i, rec in enumerate(model1_recovery[:3]):  # Show first 3
            status = "RECOVERED" if rec['recovered'] else "NOT RECOVERED"
            print(f"  Collapse at step {rec['collapse_step']}: {status}")
            if rec['recovered']:
                print(f"    → Recovery time: {rec['recovery_time']} episodes")
            print(f"    → Max recovery: {rec['recovery_percentage']:.1f}% of baseline")

    if model2_collapses:
        print("\n" + "="*70)
        print("RECOVERY ANALYSIS - Model 2")
        print("="*70)

        model2_recovery = analyze_collapse_recovery(model2_rewards, model2_collapses)
        recovered_count = sum(1 for r in model2_recovery if r['recovered'])
        print(f"\nRecovered from {recovered_count}/{len(model2_collapses)} collapses")

        for i, rec in enumerate(model2_recovery[:3]):  # Show first 3
            status = "RECOVERED" if rec['recovered'] else "NOT RECOVERED"
            print(f"  Collapse at step {rec['collapse_step']}: {status}")
            if rec['recovered']:
                print(f"    → Recovery time: {rec['recovery_time']} episodes")
            print(f"    → Max recovery: {rec['recovery_percentage']:.1f}% of baseline")

    # Compare collapse rates
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)

    comparison = compare_collapse_rates(model1_rewards, model2_rewards,
                                       window_size=window_size, threshold=threshold)

    print(f"\nCollapse Rates (per 1000 episodes):")
    print(f"  Model 1: {comparison['ppo_collapse_rate']:.2f}")
    print(f"  Model 2: {comparison['metappo_collapse_rate']:.2f}")

    # Generate summaries
    model1_summary = generate_collapse_summary(model1_collapses)
    model2_summary = generate_collapse_summary(model2_collapses)

    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)

    print(f"\nModel 1 (Non-Tested, no L2, minibatch=4):")
    print(f"  Mean reward: {np.mean(model1_rewards):.2f} ± {np.std(model1_rewards):.2f}")
    print(f"  Min/Max: {np.min(model1_rewards):.2f} / {np.max(model1_rewards):.2f}")

    print(f"\nModel 2 (L2-Reg 1e-4, minibatch=32):")
    print(f"  Mean reward: {np.mean(model2_rewards):.2f} ± {np.std(model2_rewards):.2f}")
    print(f"  Min/Max: {np.min(model2_rewards):.2f} / {np.max(model2_rewards):.2f}")

    # Determine winner
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    model1_score = 0
    model2_score = 0

    # Criteria 1: Fewer collapses
    if len(model1_collapses) < len(model2_collapses):
        model1_score += 1
        print("\n✓ Model 1 has FEWER collapse events")
    elif len(model2_collapses) < len(model1_collapses):
        model2_score += 1
        print("\n✓ Model 2 has FEWER collapse events")
    else:
        print("\n= Both models have EQUAL collapse events")

    # Criteria 2: Better stability
    if model1_stability < model2_stability:
        model1_score += 1
        print("✓ Model 1 is MORE STABLE")
    else:
        model2_score += 1
        print("✓ Model 2 is MORE STABLE")

    # Criteria 3: Higher mean performance
    if np.mean(model1_rewards) > np.mean(model2_rewards):
        model1_score += 1
        print("✓ Model 1 has HIGHER mean performance")
    else:
        model2_score += 1
        print("✓ Model 2 has HIGHER mean performance")

    print(f"\nFinal Score: Model 1 = {model1_score}, Model 2 = {model2_score}")

    if model1_score > model2_score:
        print("\n🏆 WINNER: Model 1 (Non-Tested, no L2)")
    elif model2_score > model1_score:
        print("\n🏆 WINNER: Model 2 (L2-Reg)")
    else:
        print("\n🤝 TIE: Both models perform similarly")

    # Save detailed results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Prepare data for visualization
    model1_data = {
        'rewards': model1_rewards,
        'collapses': model1_collapses,
        'stability': model1_stability,
        'summary': model1_summary
    }

    model2_data = {
        'rewards': model2_rewards,
        'collapses': model2_collapses,
        'stability': model2_stability,
        'summary': model2_summary
    }

    # Generate visualization
    output_path = Path("collapse_analysis_comparison.png")
    plot_collapse_analysis(model1_data, model2_data, output_path)

    # Save detailed CSV report
    csv_path = Path("collapse_analysis_report.csv")
    report_data = {
        'Model': ['Non-Tested (no L2)', 'L2-Reg (1e-4)'],
        'Total_Collapses': [len(model1_collapses), len(model2_collapses)],
        'Collapse_Rate_per_1000': [comparison['ppo_collapse_rate'], comparison['metappo_collapse_rate']],
        'Stability_Score': [model1_stability, model2_stability],
        'Mean_Reward': [np.mean(model1_rewards), np.mean(model2_rewards)],
        'Std_Reward': [np.std(model1_rewards), np.std(model2_rewards)],
        'Min_Reward': [np.min(model1_rewards), np.min(model2_rewards)],
        'Max_Reward': [np.max(model1_rewards), np.max(model2_rewards)],
        'Mean_Drop_Pct': [model1_summary['mean_drop'], model2_summary['mean_drop']],
        'Max_Drop_Pct': [model1_summary['max_drop'], model2_summary['max_drop']]
    }

    report_df = pd.DataFrame(report_data)
    report_df.to_csv(csv_path, index=False)
    print(f"\nDetailed report saved to: {csv_path}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
