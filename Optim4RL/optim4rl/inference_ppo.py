#!/usr/bin/env python3
"""
PPO Policy Inference Script

This script loads a trained PPO policy and runs inference in Brax environments.
It supports both agent_param and inference_param pickle files and can automatically
detect the file type from the filename.

Features:
- Auto-detection of parameter file type (agent_param vs inference_param)
- Support for both stationary and non-stationary environments
- Visualization of inference results (episode returns, lengths, distributions)
- CSV export of results for further analysis
- CPU/GPU execution modes

By default, runs on CPU to avoid CUDA/GPU issues. Use --use_gpu to force GPU mode.

Usage Examples:
    # Basic inference with agent_param (auto-detected)
    python inference_ppo.py --policy_path ./logs/ppo_ant/1/agent_param_iter5.pickle \
                            --num_episodes 50

    # Inference with visualization
    python inference_ppo.py --policy_path ./logs/ppo_ant/1/agent_param_iter5.pickle \
                            --num_episodes 50 \
                            --visualize

    # Non-stationary environment
    python inference_ppo.py --policy_path ./logs/ppo_ant_nonstationary/1/inference_param_iter20.pickle \
                            --num_episodes 50 \
                            --non_stationary \
                            --visualize

    # Run on GPU with custom output directory
    python inference_ppo.py --policy_path ./logs/ppo_ant/1/agent_param_iter5.pickle \
                            --num_episodes 50 \
                            --visualize \
                            --output_dir ./results \
                            --use_gpu

Note:
- agent_param files don't include the observation normalizer, so observations won't be normalized during inference.
- inference_param files include both normalizer and policy for proper inference.
- For best results, use inference_param files when available.
"""

import argparse
import functools
import os
import sys

# Check if user explicitly requested GPU before imports
# Default to CPU for inference to avoid CUDA issues
if '--use_gpu' not in sys.argv:
    os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
from brax import envs
from brax.training import acting
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import specs
from envs.non_stationary_wrapper import NonStationaryDynamicsWrapper
from utils.helper import load_model_param
from components import running_statistics
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp


def run_inference(policy_path, env_name='ant', num_episodes=10, episode_length=1000,
                 non_stationary=True, task_type='ant', seed=0, param_type='auto',
                 visualize=False, output_dir=None):
    """
    Run inference with a trained policy

    Args:
        policy_path: Path to saved pickle file (inference_param or agent_param)
        env_name: Environment name
        num_episodes: Number of episodes to run
        episode_length: Maximum steps per episode
        non_stationary: Whether to use non-stationary dynamics
        task_type: Task type for non-stationary wrapper
        seed: Random seed
        param_type: Type of parameter file ('auto', 'inference', or 'agent')
                    'auto' will detect from filename
        visualize: Whether to generate visualization graphs
        output_dir: Directory to save visualization plots (uses same dir as policy if None)

    Returns:
        episode_returns: List of episode returns
        episode_lengths: List of episode lengths
    """
    # Print device info
    print(f"JAX devices: {jax.devices()}")
    print(f"Loading policy from: {policy_path}")

    # Auto-detect parameter type from filename if set to 'auto'
    if param_type == 'auto':
        filename = os.path.basename(policy_path)
        if 'agent_param' in filename:
            param_type = 'agent'
        elif 'inference_param' in filename:
            param_type = 'inference'
        else:
            # Default to inference for backward compatibility
            param_type = 'inference'
            print(f"Warning: Could not detect parameter type from filename. Defaulting to 'inference'.")

    print(f"Parameter type: {param_type}")

    # Load parameters based on type
    loaded_params = load_model_param(policy_path)

    if param_type == 'inference':
        # inference_param format: (normalizer_param, policy_param)
        normalizer_param, policy_param = loaded_params
    elif param_type == 'agent':
        # agent_param format: PPONetworkParams(policy, value)
        # We only need the policy part for inference
        agent_param = loaded_params
        policy_param = agent_param.policy

        # Create a new normalizer_param since agent_param doesn't include it
        # We'll initialize it with default values (will be in non-normalized state)
        print("Warning: agent_param doesn't include normalizer. Creating new normalizer (observations won't be normalized).")

        # Create environment temporarily to get observation size
        temp_env = envs.get_environment(env_name=env_name, backend='spring')
        obs_size = temp_env.observation_size
        normalizer_param = running_statistics.init_state(
            specs.Array((obs_size,), jnp.dtype('float32'))
        )
        del temp_env
    else:
        raise ValueError(f"param_type must be 'inference' or 'agent', got: {param_type}")

    # Create environment
    backends = ['generalized', 'positional', 'spring']
    env = envs.get_environment(env_name=env_name, backend=backends[2])

    if non_stationary:
        env = NonStationaryDynamicsWrapper(env, task_type=task_type, change_mode='continuous')
        print(f"Non-stationary dynamics applied: task_type={task_type}")

    # Wrap environment for evaluation (needed for proper episode handling)
    eval_env = envs.training.wrap(
        env,
        episode_length=episode_length,
        action_repeat=1,
        randomization_fn=None
    )

    # Create policy network
    ppo_network = ppo_networks.make_ppo_networks(
        eval_env.observation_size,
        eval_env.action_size,
        preprocess_observations_fn=lambda x, _: x  # Normalization will be handled by normalizer_param
    )

    # Create inference function
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    # Run episodes sequentially for detailed per-episode statistics
    key = jax.random.PRNGKey(seed)
    episode_returns = []
    episode_lengths = []

    print(f"\nRunning {num_episodes} episodes...")
    for episode in range(num_episodes):
        key, eval_key = jax.random.split(key)
        single_evaluator = acting.Evaluator(
            eval_env,
            functools.partial(make_policy, deterministic=True),
            num_eval_envs=1,
            episode_length=episode_length,
            action_repeat=1,
            key=eval_key
        )

        ep_metrics = single_evaluator.run_evaluation(
            (normalizer_param, policy_param),
            training_metrics={}
        )

        ep_return = float(ep_metrics['eval/episode_reward'])
        ep_length = float(ep_metrics.get('eval/episode_length', episode_length))

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)

        print(f"Episode {episode+1}/{num_episodes}: Return={ep_return:.2f}, Length={ep_length:.0f}")

    # Print statistics
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    mean_length = np.mean(episode_lengths)

    print(f"\n{'='*60}")
    print(f"Inference Results:")
    print(f"  Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"  Mean Episode Length: {mean_length:.1f}")
    print(f"  Min Return: {np.min(episode_returns):.2f}")
    print(f"  Max Return: {np.max(episode_returns):.2f}")
    print(f"{'='*60}\n")

    # Generate visualization if requested
    if visualize:
        if output_dir is None:
            # Use the same directory as the policy file
            output_dir = os.path.dirname(policy_path)

        os.makedirs(output_dir, exist_ok=True)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Inference Results - {env_name.upper()}', fontsize=16, fontweight='bold')

        # Plot 1: Episode Returns over Episodes
        axes[0, 0].plot(range(1, num_episodes + 1), episode_returns, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].axhline(y=mean_return, color='r', linestyle='--', linewidth=1.5, label=f'Mean: {mean_return:.2f}')
        axes[0, 0].fill_between(range(1, num_episodes + 1),
                                mean_return - std_return,
                                mean_return + std_return,
                                alpha=0.2, color='r', label=f'±1 Std: {std_return:.2f}')
        axes[0, 0].set_xlabel('Episode', fontsize=12)
        axes[0, 0].set_ylabel('Return', fontsize=12)
        axes[0, 0].set_title('Episode Returns', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Episode Lengths over Episodes
        axes[0, 1].plot(range(1, num_episodes + 1), episode_lengths, 'g-s', linewidth=2, markersize=6)
        axes[0, 1].axhline(y=mean_length, color='r', linestyle='--', linewidth=1.5, label=f'Mean: {mean_length:.1f}')
        axes[0, 1].set_xlabel('Episode', fontsize=12)
        axes[0, 1].set_ylabel('Length', fontsize=12)
        axes[0, 1].set_title('Episode Lengths', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Distribution of Returns (Histogram)
        axes[1, 0].hist(episode_returns, bins=min(10, num_episodes//2), color='skyblue', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=mean_return, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.2f}')
        axes[1, 0].set_xlabel('Return', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Return Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Plot 4: Summary Statistics (Text Box)
        axes[1, 1].axis('off')
        summary_text = f"""
        Summary Statistics
        {'='*30}

        Episodes:           {num_episodes}
        Environment:        {env_name}
        Non-stationary:     {non_stationary}
        Task Type:          {task_type}
        Parameter Type:     {param_type}

        {'='*30}
        Return Statistics
        {'='*30}
        Mean:               {mean_return:.2f}
        Std Dev:            {std_return:.2f}
        Min:                {np.min(episode_returns):.2f}
        Max:                {np.max(episode_returns):.2f}
        Median:             {np.median(episode_returns):.2f}

        {'='*30}
        Length Statistics
        {'='*30}
        Mean:               {mean_length:.1f}
        Min:                {np.min(episode_lengths):.0f}
        Max:                {np.max(episode_lengths):.0f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(output_dir, f'inference_results_{env_name}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {plot_path}")

        # Also save data to CSV for further analysis
        csv_path = os.path.join(output_dir, f'inference_data_{env_name}.csv')
        import pandas as pd
        df = pd.DataFrame({
            'Episode': range(1, num_episodes + 1),
            'Return': episode_returns,
            'Length': episode_lengths
        })
        df.to_csv(csv_path, index=False)
        print(f"Data saved to: {csv_path}")

        plt.close()

    return episode_returns, episode_lengths


def main():
    parser = argparse.ArgumentParser(description='PPO Policy Inference',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
Examples:
  # Using agent_param (auto-detected)
  python inference_ppo.py --policy_path ./logs/ppo_ant/1/agent_param_iter5.pickle --num_episodes 50

  # Using inference_param (auto-detected) with visualization
  python inference_ppo.py --policy_path ./logs/ppo_ant/1/inference_param_iter20.pickle \\
                          --num_episodes 50 --visualize

  # Manually specify parameter type
  python inference_ppo.py --policy_path ./logs/ppo_ant/1/agent_param_iter5.pickle \\
                          --param_type agent --num_episodes 50 --visualize
""")
    parser.add_argument('--policy_path', type=str, required=True,
                       help='Path to agent_param or inference_param pickle file')
    parser.add_argument('--env_name', type=str, default='ant',
                       help='Environment name (default: ant)')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Number of episodes to run (default: 10)')
    parser.add_argument('--episode_length', type=int, default=1000,
                       help='Maximum episode length (default: 1000)')
    parser.add_argument('--non_stationary', action='store_true', default=False,
                       help='Use non-stationary dynamics (default: False)')
    parser.add_argument('--task_type', type=str, default='ant',
                       help='Task type for non-stationary wrapper (default: ant)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--param_type', type=str, default='auto',
                       choices=['auto', 'agent', 'inference'],
                       help='Parameter file type: auto (detect from filename), agent, or inference (default: auto)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization graphs and save results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualization plots (default: same as policy file)')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU instead of CPU (default: CPU)')

    args = parser.parse_args()

    run_inference(
        policy_path=args.policy_path,
        env_name=args.env_name,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        non_stationary=args.non_stationary,
        task_type=args.task_type,
        seed=args.seed,
        param_type=args.param_type,
        visualize=args.visualize,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
