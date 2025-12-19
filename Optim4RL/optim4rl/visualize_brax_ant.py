#!/usr/bin/env python3
"""
Brax Ant Visualization Script

Visualizes trained PPO policies in Brax Ant environment with HTML rendering.
Supports both stationary and non-stationary environments.

Usage:
    python visualize_brax_ant.py --policy_path ./logs/ppo_ant/1/inference_param_iter20.pickle \
                                  --output_dir ./videos \
                                  --num_episodes 3 \
                                  --episode_length 1000

    # With non-stationary dynamics
    python visualize_brax_ant.py --policy_path ./logs/ppo_ant_non_tested/1/inference_param_iter5.pickle \
                                  --non_stationary \
                                  --num_episodes 3
"""

import argparse
import functools
import os
import sys
from pathlib import Path

# Default to CPU to avoid CUDA issues
if '--use_gpu' not in sys.argv:
    os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
from brax import envs
from brax.io import html
from brax.training import acting
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import specs
from envs.non_stationary_wrapper import NonStationaryDynamicsWrapper
from utils.helper import load_model_param
from components import running_statistics
import numpy as np


def create_video(policy_path, env_name='ant', num_episodes=3, episode_length=1000,
                non_stationary=False, task_type='ant', seed=0, param_type='auto',
                output_dir=None, framerate=60):
    """
    Create HTML visualization of policy execution in Brax environment.

    Args:
        policy_path: Path to saved pickle file (inference_param or agent_param)
        env_name: Environment name (default: ant)
        num_episodes: Number of episodes to visualize
        episode_length: Maximum steps per episode
        non_stationary: Whether to use non-stationary dynamics
        task_type: Task type for non-stationary wrapper
        seed: Random seed
        param_type: Type of parameter file ('auto', 'inference', or 'agent')
        output_dir: Directory to save HTML files
        framerate: Video framerate (default: 60 fps)

    Returns:
        List of paths to generated HTML files
    """
    print("="*70)
    print("BRAX ANT VISUALIZATION")
    print("="*70)
    print(f"\nPolicy: {policy_path}")
    print(f"Environment: {env_name}")
    print(f"Non-stationary: {non_stationary}")
    print(f"Episodes: {num_episodes}")
    print(f"Episode length: {episode_length}")
    print(f"JAX devices: {jax.devices()}")

    # Auto-detect parameter type from filename
    if param_type == 'auto':
        filename = os.path.basename(policy_path)
        if 'agent_param' in filename:
            param_type = 'agent'
        elif 'inference_param' in filename:
            param_type = 'inference'
        else:
            param_type = 'inference'
            print(f"Warning: Could not detect parameter type. Defaulting to 'inference'.")

    print(f"Parameter type: {param_type}")

    # Load parameters
    print("\nLoading policy parameters...")
    loaded_params = load_model_param(policy_path)

    if param_type == 'inference':
        normalizer_param, policy_param = loaded_params
    elif param_type == 'agent':
        agent_param = loaded_params
        policy_param = agent_param.policy

        # Create normalizer_param
        print("Warning: agent_param doesn't include normalizer. Creating new normalizer.")
        temp_env = envs.get_environment(env_name=env_name, backend='spring')
        obs_size = temp_env.observation_size
        normalizer_param = running_statistics.init_state(
            specs.Array((obs_size,), jnp.dtype('float32'))
        )
        del temp_env
    else:
        raise ValueError(f"param_type must be 'inference' or 'agent', got: {param_type}")

    # Create environment
    print("\nCreating Brax environment...")
    base_env = envs.get_environment(env_name=env_name, backend='spring')

    if non_stationary:
        base_env = NonStationaryDynamicsWrapper(base_env, task_type=task_type, change_mode='continuous')
        print(f"Non-stationary dynamics applied: task_type={task_type}")

    # Use base environment directly for visualization (no training wrapper)
    # Training wrapper causes issues with key shapes in reset
    env = base_env

    # Create policy network
    print("Creating policy network...")
    ppo_network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=lambda x, _: x
    )

    # Create inference function
    # make_policy takes params and returns a policy function
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    # Create policy function with our parameters
    policy_fn = make_policy((normalizer_param, policy_param))

    # Setup output directory
    if output_dir is None:
        output_dir = Path(policy_path).parent / "videos"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate videos for each episode
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    key = jax.random.PRNGKey(seed)
    html_files = []

    for episode_idx in range(num_episodes):
        print(f"\nEpisode {episode_idx + 1}/{num_episodes}:")

        # Reset environment
        key, reset_key = jax.random.split(key)
        state = env.reset(reset_key)

        # Rollout trajectory
        states = [state]
        total_reward = 0.0

        print("  Rolling out trajectory...")

        # First step (JIT compilation happens here)
        print("    Step 0 (JIT compiling, this may take 1-2 minutes)...", flush=True)
        key, action_key = jax.random.split(key)
        action, _ = policy_fn(state.obs, action_key)
        state = env.step(state, action)
        states.append(state)
        total_reward += float(state.reward)
        print("    JIT compilation done!", flush=True)

        # Remaining steps (should be faster)
        print(f"    Running remaining {episode_length-1} steps...", end='', flush=True)
        for step in range(1, episode_length):
            # Get action from policy
            key, action_key = jax.random.split(key)
            action, _ = policy_fn(state.obs, action_key)

            # Step environment
            state = env.step(state, action)
            states.append(state)
            total_reward += float(state.reward)

            # Progress indicator
            if step > 0 and step % 10 == 0:
                print(f".", end='', flush=True)

        print(f" Completed {len(states)-1} steps")
        print(f"  Total reward: {total_reward:.2f}")

        # Generate HTML visualization
        print("  Generating HTML...", end='', flush=True)

        # Extract QP (position/velocity) states for visualization
        qp_states = [s.pipeline_state for s in states]

        # Get the base environment (unwrap if wrapped)
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        # Create HTML (try different approaches for different Brax versions)
        try:
            # Try with tree_replace (newer Brax)
            html_string = html.render(
                base_env.sys.tree_replace({'opt.timestep': base_env.dt}),
                qp_states,
                height=480,
                colab=False
            )
        except AttributeError:
            # Try direct sys (older Brax or different API)
            try:
                html_string = html.render(
                    base_env.sys,
                    qp_states,
                    height=480,
                    colab=False
                )
            except Exception as e:
                print(f"\nError generating HTML: {e}")
                print("Trying without sys modifications...")
                try:
                    # Last resort: use sys directly
                    html_string = html.render(
                        base_env.sys,
                        qp_states
                    )
                except Exception as e2:
                    print(f"Error: {e2}")
                    print("Skipping HTML generation for this episode")
                    continue

        # Save HTML file
        policy_name = Path(policy_path).stem
        html_filename = f"{policy_name}_episode{episode_idx+1}_reward{total_reward:.0f}.html"
        html_path = output_dir / html_filename

        with open(html_path, 'w') as f:
            f.write(html_string)

        html_files.append(str(html_path))
        print(f" Saved to {html_path}")

    # Create index HTML with all episodes
    print("\nCreating index page...")
    index_html = generate_index_page(html_files, policy_path, non_stationary)
    index_path = output_dir / "index.html"

    with open(index_path, 'w') as f:
        f.write(index_html)

    print(f"Index page saved to: {index_path}")

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nGenerated {len(html_files)} episode visualizations")
    print(f"Output directory: {output_dir}")
    print(f"\nTo view:")
    print(f"  Open in browser: {index_path}")
    print(f"  Or individual files in: {output_dir}/")

    return html_files


def generate_index_page(html_files, policy_path, non_stationary):
    """Generate index HTML page with links to all episode visualizations."""
    policy_name = Path(policy_path).stem
    episodes_html = ""

    for i, html_file in enumerate(html_files, 1):
        filename = Path(html_file).name
        episodes_html += f"""
        <li>
            <a href="{filename}" target="_blank">{filename}</a>
        </li>
        """

    index_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Brax Ant Visualization - {policy_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .info {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .info-item {{
            margin: 5px 0;
        }}
        .label {{
            font-weight: bold;
            color: #2c3e50;
        }}
        ul {{
            list-style-type: none;
            padding: 0;
        }}
        li {{
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            border-radius: 3px;
        }}
        a {{
            color: #2980b9;
            text-decoration: none;
            font-size: 16px;
        }}
        a:hover {{
            color: #3498db;
            text-decoration: underline;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Brax Ant Visualization</h1>

        <div class="info">
            <div class="info-item">
                <span class="label">Policy:</span> {policy_name}
            </div>
            <div class="info-item">
                <span class="label">Environment:</span> Brax Ant
            </div>
            <div class="info-item">
                <span class="label">Non-stationary:</span> {'Yes' if non_stationary else 'No'}
            </div>
            <div class="info-item">
                <span class="label">Episodes:</span> {len(html_files)}
            </div>
        </div>

        <h2>📹 Episode Visualizations</h2>
        <p>Click on an episode to view the Brax simulation in a new tab:</p>

        <ul>
            {episodes_html}
        </ul>

        <div class="footer">
            Generated by Optim4RL Brax Visualization Tool
        </div>
    </div>
</body>
</html>
    """

    return index_html


def main():
    parser = argparse.ArgumentParser(
        description='Visualize trained PPO policies in Brax Ant environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic visualization
  python visualize_brax_ant.py --policy_path ./logs/ppo_ant_non_tested/1/inference_param_iter5.pickle

  # Non-stationary environment with 5 episodes
  python visualize_brax_ant.py --policy_path ./logs/ppo_ant_non_tested/1/inference_param_iter5.pickle \\
                                --non_stationary \\
                                --num_episodes 5

  # Custom output directory
  python visualize_brax_ant.py --policy_path ./logs/ppo_ant_non_tested/1/inference_param_iter5.pickle \\
                                --output_dir ./my_videos \\
                                --num_episodes 3
        """
    )

    parser.add_argument('--policy_path', type=str, required=True,
                       help='Path to agent_param or inference_param pickle file')
    parser.add_argument('--env_name', type=str, default='ant',
                       help='Environment name (default: ant)')
    parser.add_argument('--num_episodes', type=int, default=3,
                       help='Number of episodes to visualize (default: 3)')
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
                       help='Parameter file type (default: auto)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save HTML files (default: policy_path/videos)')
    parser.add_argument('--framerate', type=int, default=60,
                       help='Video framerate (default: 60 fps)')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU instead of CPU (default: CPU)')

    args = parser.parse_args()

    # Create videos
    html_files = create_video(
        policy_path=args.policy_path,
        env_name=args.env_name,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        non_stationary=args.non_stationary,
        task_type=args.task_type,
        seed=args.seed,
        param_type=args.param_type,
        output_dir=args.output_dir,
        framerate=args.framerate
    )


if __name__ == '__main__':
    main()
