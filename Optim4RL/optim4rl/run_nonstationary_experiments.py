#!/usr/bin/env python3
# Copyright 2024 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Non-Stationary Task Execution Script

This script provides an easy interface to run all non-stationary experiments
for policy collapse analysis.

Usage:
    # Run all experiments
    python run_nonstationary_experiments.py --all

    # Run specific experiment
    python run_nonstationary_experiments.py --experiment ppo_ant

    # List available experiments
    python run_nonstationary_experiments.py --list

    # Dry run (show commands without executing)
    python run_nonstationary_experiments.py --all --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

# Define all non-stationary experiments
EXPERIMENTS = {
    'ppo_ant': {
        'config': './configs/ppo_ant_nonstationary.json',
        'description': 'PPO on Ant non-stationary task',
        'priority': 1
    },
    'ppo_humanoid': {
        'config': './configs/ppo_humanoid_nonstationary.json',
        'description': 'PPO on Humanoid non-stationary task',
        'priority': 2
    },
    'metappo_ant': {
        'config': './configs/metap_ppo_ant_nonstationary.json',
        'description': 'MetaPPO on Ant non-stationary task',
        'priority': 3
    },
    'metappo_humanoid': {
        'config': './configs/metap_ppo_humanoid_nonstationary.json',
        'description': 'MetaPPO on Humanoid non-stationary task',
        'priority': 4
    }
}

# Baseline experiments (for comparison)
BASELINE_EXPERIMENTS = {
    'ppo_ant_baseline': {
        'config': './configs/ppo_ant_adam.json',
        'description': 'PPO on Ant stationary task (baseline)',
        'priority': 5
    }
}


def run_experiment(config_file: str, config_idx: int = 1, dry_run: bool = False) -> int:
    """
    Run a single experiment.

    Args:
        config_file: Path to configuration file
        config_idx: Configuration index
        dry_run: If True, only print command without executing

    Returns:
        Return code (0 for success)
    """
    cmd = f"python main.py --config_file {config_file} --config_idx {config_idx}"

    print(f"\n{'='*80}")
    print(f"Running: {cmd}")
    print(f"{'='*80}\n")

    if dry_run:
        print("[DRY RUN] Command not executed")
        return 0

    try:
        result = subprocess.run(cmd, shell=True, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return 1


def list_experiments():
    """Print list of available experiments."""
    print("\n" + "="*80)
    print("AVAILABLE NON-STATIONARY EXPERIMENTS")
    print("="*80 + "\n")

    print("Non-Stationary Experiments:")
    print("-" * 80)
    for name, info in sorted(EXPERIMENTS.items(), key=lambda x: x[1]['priority']):
        print(f"  {name:20s} - {info['description']}")
        print(f"  {'':20s}   Config: {info['config']}")
        print()

    print("\nBaseline Experiments (for comparison):")
    print("-" * 80)
    for name, info in BASELINE_EXPERIMENTS.items():
        print(f"  {name:20s} - {info['description']}")
        print(f"  {'':20s}   Config: {info['config']}")
        print()


def run_all_experiments(dry_run: bool = False, include_baseline: bool = False):
    """
    Run all non-stationary experiments.

    Args:
        dry_run: If True, only print commands
        include_baseline: If True, also run baseline experiments
    """
    experiments = dict(EXPERIMENTS)
    if include_baseline:
        experiments.update(BASELINE_EXPERIMENTS)

    total = len(experiments)
    completed = 0
    failed = []

    print("\n" + "="*80)
    print(f"RUNNING {total} EXPERIMENTS")
    print("="*80)

    for name, info in sorted(experiments.items(), key=lambda x: x[1]['priority']):
        print(f"\n[{completed + 1}/{total}] Starting: {name}")
        print(f"Description: {info['description']}")

        returncode = run_experiment(info['config'], config_idx=1, dry_run=dry_run)

        if returncode == 0:
            completed += 1
            print(f"✓ {name} completed successfully")
        else:
            failed.append(name)
            print(f"✗ {name} failed with return code {returncode}")

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total: {total}")
    print(f"Completed: {completed}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed experiments:")
        for name in failed:
            print(f"  - {name}")

    print()


def check_configs():
    """Check if all config files exist."""
    print("\n" + "="*80)
    print("CHECKING CONFIGURATION FILES")
    print("="*80 + "\n")

    all_experiments = dict(EXPERIMENTS)
    all_experiments.update(BASELINE_EXPERIMENTS)

    missing = []
    for name, info in all_experiments.items():
        config_path = Path(info['config'])
        if config_path.exists():
            print(f"✓ {name:20s} - {config_path}")
        else:
            print(f"✗ {name:20s} - {config_path} (MISSING)")
            missing.append(name)

    if missing:
        print(f"\n⚠ Warning: {len(missing)} config files are missing!")
        print("Missing configs for:", ", ".join(missing))
        return False
    else:
        print(f"\n✓ All {len(all_experiments)} config files found!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Run non-stationary task experiments for policy collapse analysis'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all non-stationary experiments'
    )

    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Include baseline (stationary) experiments'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        choices=list(EXPERIMENTS.keys()) + list(BASELINE_EXPERIMENTS.keys()),
        help='Run specific experiment'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available experiments'
    )

    parser.add_argument(
        '--check',
        action='store_true',
        help='Check if all config files exist'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )

    parser.add_argument(
        '--config-idx',
        type=int,
        default=1,
        help='Configuration index (default: 1)'
    )

    args = parser.parse_args()

    # Handle list command
    if args.list:
        list_experiments()
        return 0

    # Handle check command
    if args.check:
        if check_configs():
            return 0
        else:
            return 1

    # Handle all experiments
    if args.all:
        run_all_experiments(dry_run=args.dry_run, include_baseline=args.baseline)
        return 0

    # Handle single experiment
    if args.experiment:
        all_experiments = dict(EXPERIMENTS)
        all_experiments.update(BASELINE_EXPERIMENTS)

        if args.experiment not in all_experiments:
            print(f"Error: Unknown experiment '{args.experiment}'")
            list_experiments()
            return 1

        info = all_experiments[args.experiment]
        print(f"\nRunning: {args.experiment}")
        print(f"Description: {info['description']}")

        returncode = run_experiment(info['config'], config_idx=args.config_idx, dry_run=args.dry_run)
        return returncode

    # No action specified
    parser.print_help()
    print("\n💡 Tip: Use --list to see available experiments")
    print("💡 Tip: Use --check to verify config files")
    print("💡 Tip: Use --all to run all experiments")
    return 0


if __name__ == '__main__':
    sys.exit(main())
