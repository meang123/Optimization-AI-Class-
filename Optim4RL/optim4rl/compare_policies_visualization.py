#!/usr/bin/env python3
"""
Compare Two Policies in Brax Visualization

Generates side-by-side visualizations of two different policies for comparison.

Usage:
    python compare_policies_visualization.py \\
        --policy1 ./logs/ppo_ant_non_tested/1/inference_param_iter5.pickle \\
        --policy2 ./logs/ppo_ant_nonstationary_test_learned_l2reg/1/inference_param_iter5.pickle \\
        --num_episodes 3
"""

import argparse
import os
import sys
from pathlib import Path

# Default to CPU
if '--use_gpu' not in sys.argv:
    os.environ['JAX_PLATFORMS'] = 'cpu'

from visualize_brax_ant import create_video


def compare_policies(policy1_path, policy2_path, policy1_name=None, policy2_name=None,
                    num_episodes=3, episode_length=1000, non_stationary=False,
                    output_dir=None, seed=0):
    """
    Generate visualizations for two policies and create comparison page.

    Args:
        policy1_path: Path to first policy
        policy2_path: Path to second policy
        policy1_name: Display name for policy 1 (default: filename)
        policy2_name: Display name for policy 2 (default: filename)
        num_episodes: Number of episodes to visualize per policy
        episode_length: Maximum episode length
        non_stationary: Whether to use non-stationary dynamics
        output_dir: Output directory for all files
        seed: Random seed
    """
    print("="*70)
    print("POLICY COMPARISON VISUALIZATION")
    print("="*70)

    # Setup output directory
    if output_dir is None:
        output_dir = Path("policy_comparison_videos")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    policy1_dir = output_dir / "policy1"
    policy2_dir = output_dir / "policy2"

    # Get policy names
    if policy1_name is None:
        policy1_name = Path(policy1_path).stem
    if policy2_name is None:
        policy2_name = Path(policy2_path).stem

    print(f"\nPolicy 1: {policy1_name}")
    print(f"  Path: {policy1_path}")
    print(f"\nPolicy 2: {policy2_name}")
    print(f"  Path: {policy2_path}")
    print(f"\nOutput directory: {output_dir}")

    # Generate visualizations for policy 1
    print("\n" + "="*70)
    print("GENERATING POLICY 1 VISUALIZATIONS")
    print("="*70)

    html_files_1 = create_video(
        policy_path=policy1_path,
        num_episodes=num_episodes,
        episode_length=episode_length,
        non_stationary=non_stationary,
        seed=seed,
        output_dir=policy1_dir
    )

    # Generate visualizations for policy 2
    print("\n" + "="*70)
    print("GENERATING POLICY 2 VISUALIZATIONS")
    print("="*70)

    html_files_2 = create_video(
        policy_path=policy2_path,
        num_episodes=num_episodes,
        episode_length=episode_length,
        non_stationary=non_stationary,
        seed=seed,
        output_dir=policy2_dir
    )

    # Create comparison index page
    print("\n" + "="*70)
    print("CREATING COMPARISON PAGE")
    print("="*70)

    comparison_html = generate_comparison_page(
        policy1_name, policy2_name,
        html_files_1, html_files_2,
        policy1_dir, policy2_dir,
        non_stationary
    )

    comparison_path = output_dir / "comparison.html"
    with open(comparison_path, 'w') as f:
        f.write(comparison_html)

    print(f"\nComparison page saved to: {comparison_path}")

    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nGenerated:")
    print(f"  Policy 1: {len(html_files_1)} episodes in {policy1_dir}/")
    print(f"  Policy 2: {len(html_files_2)} episodes in {policy2_dir}/")
    print(f"\nTo view:")
    print(f"  Open in browser: {comparison_path}")


def generate_comparison_page(policy1_name, policy2_name, html_files_1, html_files_2,
                            policy1_dir, policy2_dir, non_stationary):
    """Generate comparison HTML page."""
    # Create episode comparison table
    episodes_html = ""

    for i in range(max(len(html_files_1), len(html_files_2))):
        episode_num = i + 1

        # Policy 1 link
        if i < len(html_files_1):
            file1 = Path(html_files_1[i])
            rel_path1 = f"policy1/{file1.name}"
            link1 = f'<a href="{rel_path1}" target="_blank">{file1.name}</a>'
        else:
            link1 = '<span style="color: #999;">N/A</span>'

        # Policy 2 link
        if i < len(html_files_2):
            file2 = Path(html_files_2[i])
            rel_path2 = f"policy2/{file2.name}"
            link2 = f'<a href="{rel_path2}" target="_blank">{file2.name}</a>'
        else:
            link2 = '<span style="color: #999;">N/A</span>'

        episodes_html += f"""
        <tr>
            <td style="text-align: center; font-weight: bold;">Episode {episode_num}</td>
            <td>{link1}</td>
            <td>{link2}</td>
        </tr>
        """

    comparison_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Policy Comparison - Brax Ant</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            font-size: 36px;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            font-size: 18px;
            margin-bottom: 30px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 30px 0;
        }}
        .policy-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #3498db;
        }}
        .policy-card h2 {{
            color: #2c3e50;
            margin-top: 0;
            font-size: 24px;
        }}
        .policy-info {{
            margin: 10px 0;
            color: #34495e;
        }}
        .label {{
            font-weight: bold;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 30px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-size: 16px;
        }}
        td {{
            padding: 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        a {{
            color: #2980b9;
            text-decoration: none;
            font-size: 14px;
            transition: color 0.3s;
        }}
        a:hover {{
            color: #3498db;
            text-decoration: underline;
        }}
        .comparison-note {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .comparison-note p {{
            margin: 5px 0;
            color: #856404;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 14px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-value {{
            color: #2c3e50;
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Policy Comparison</h1>
        <div class="subtitle">Brax Ant Environment Visualization</div>

        <div class="info-grid">
            <div class="policy-card" style="border-left-color: #3498db;">
                <h2>📘 Policy 1</h2>
                <div class="policy-info">
                    <span class="label">Name:</span> {policy1_name}
                </div>
                <div class="policy-info">
                    <span class="label">Episodes:</span> {len(html_files_1)}
                </div>
                <div class="policy-info">
                    <span class="label">Location:</span> policy1/
                </div>
            </div>

            <div class="policy-card" style="border-left-color: #e74c3c;">
                <h2>📕 Policy 2</h2>
                <div class="policy-info">
                    <span class="label">Name:</span> {policy2_name}
                </div>
                <div class="policy-info">
                    <span class="label">Episodes:</span> {len(html_files_2)}
                </div>
                <div class="policy-info">
                    <span class="label">Location:</span> policy2/
                </div>
            </div>
        </div>

        <div class="comparison-note">
            <p><strong>📌 Note:</strong> Click on episode links below to open side-by-side visualizations in new tabs.</p>
            <p><strong>Environment:</strong> Brax Ant {'(Non-stationary dynamics)' if non_stationary else '(Stationary)'}</p>
            <p><strong>Tip:</strong> Open multiple episodes to compare behavior across different scenarios.</p>
        </div>

        <h2 style="color: #2c3e50; margin-top: 40px;">📹 Episode Comparisons</h2>

        <table>
            <thead>
                <tr>
                    <th style="width: 15%;">Episode</th>
                    <th style="width: 42.5%;">Policy 1: {policy1_name}</th>
                    <th style="width: 42.5%;">Policy 2: {policy2_name}</th>
                </tr>
            </thead>
            <tbody>
                {episodes_html}
            </tbody>
        </table>

        <div class="footer">
            <p>Generated by Optim4RL Policy Comparison Tool</p>
            <p style="font-size: 12px; margin-top: 10px;">
                Brax Physics Simulation | JAX-based Reinforcement Learning
            </p>
        </div>
    </div>
</body>
</html>
    """

    return comparison_html


def main():
    parser = argparse.ArgumentParser(
        description='Compare two policies with Brax visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two policies
  python compare_policies_visualization.py \\
      --policy1 ./logs/ppo_ant_non_tested/1/inference_param_iter5.pickle \\
      --policy2 ./logs/ppo_ant_nonstationary_test_learned_l2reg/1/inference_param_iter5.pickle \\
      --num_episodes 3

  # With custom names and non-stationary environment
  python compare_policies_visualization.py \\
      --policy1 ./logs/model1/inference_param.pickle \\
      --policy2 ./logs/model2/inference_param.pickle \\
      --policy1_name "No L2 Regularization" \\
      --policy2_name "With L2 Regularization" \\
      --non_stationary \\
      --num_episodes 5
        """
    )

    parser.add_argument('--policy1', type=str, required=True,
                       help='Path to first policy pickle file')
    parser.add_argument('--policy2', type=str, required=True,
                       help='Path to second policy pickle file')
    parser.add_argument('--policy1_name', type=str, default=None,
                       help='Display name for policy 1 (default: filename)')
    parser.add_argument('--policy2_name', type=str, default=None,
                       help='Display name for policy 2 (default: filename)')
    parser.add_argument('--num_episodes', type=int, default=3,
                       help='Number of episodes per policy (default: 3)')
    parser.add_argument('--episode_length', type=int, default=1000,
                       help='Maximum episode length (default: 1000)')
    parser.add_argument('--non_stationary', action='store_true', default=False,
                       help='Use non-stationary dynamics (default: False)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ./policy_comparison_videos)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU instead of CPU (default: CPU)')

    args = parser.parse_args()

    compare_policies(
        policy1_path=args.policy1,
        policy2_path=args.policy2,
        policy1_name=args.policy1_name,
        policy2_name=args.policy2_name,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        non_stationary=args.non_stationary,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
