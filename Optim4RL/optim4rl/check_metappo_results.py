#!/usr/bin/env python3
"""
Quick script to check MetaPPO training results and test performance.

This script helps understand:
1. What MetaPPO training produces (meta-parameters)
2. How to test learned optimizers
3. Compare with baseline PPO performance
"""

import os
import pickle
from pathlib import Path

def check_metappo_training(exp_name, config_idx=1):
    """Check MetaPPO training results."""
    log_dir = Path(f'./logs/{exp_name}/{config_idx}')

    print(f"\n{'='*60}")
    print(f"Checking: {exp_name}")
    print(f"{'='*60}")

    # Check if training completed
    log_file = log_dir / 'log.txt'
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Get last few lines
            print("\nLast 5 lines of log:")
            for line in lines[-5:]:
                print(f"  {line.rstrip()}")

    # Check parameter file
    param_file = log_dir / 'param.pickle'
    if param_file.exists():
        file_size = param_file.stat().st_size / 1024  # KB
        print(f"\n✓ Learned optimizer parameters saved: {file_size:.2f} KB")
        print(f"  Path: {param_file}")

        # Try to load and inspect
        try:
            with open(param_file, 'rb') as f:
                params = pickle.load(f)
            print(f"  Parameter structure: {type(params)}")
            if hasattr(params, 'keys'):
                print(f"  Keys: {list(params.keys())[:5]}...")
        except Exception as e:
            print(f"  Could not inspect params: {e}")
    else:
        print("\n✗ No parameter file found")

    # Check if test results exist
    test_file = log_dir / 'result_Test.feather'
    if test_file.exists():
        print(f"\n✓ Test results exist (this is unusual for MetaPPO training)")
    else:
        print(f"\n→ No test results (expected for MetaPPO training)")
        print(f"  Need to run separate test with learned optimizer")

def check_test_results(exp_name, config_idx=1):
    """Check test experiment results."""
    log_dir = Path(f'./logs/{exp_name}/{config_idx}')
    test_file = log_dir / 'result_Test.feather'

    print(f"\n{'='*60}")
    print(f"Checking TEST results: {exp_name}")
    print(f"{'='*60}")

    if test_file.exists():
        file_size = test_file.stat().st_size / 1024  # KB
        print(f"\n✓ Test results available: {file_size:.2f} KB")
        print(f"  Path: {test_file}")

        # Try to read with pandas if available
        try:
            import pandas as pd
            df = pd.read_feather(test_file)
            print(f"\n  Episodes: {len(df)}")
            print(f"  Mean Return: {df['Return'].mean():.2f} ± {df['Return'].std():.2f}")
            print(f"  Last 100 eps: {df['Return'][-100:].mean():.2f} ± {df['Return'][-100:].std():.2f}")
        except ImportError:
            print("\n  (Install pandas to see detailed statistics)")
        except Exception as e:
            print(f"\n  Could not read results: {e}")
    else:
        print(f"\n✗ No test results found")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MetaPPO Results Checker")
    print("="*60)

    # Check MetaPPO training experiments
    print("\n\n1. METAPPO TRAINING (learns optimizer parameters)")
    print("-" * 60)

    meta_experiments = [
        'meta_ppo_ant_nonstationary_learned_lowmem',
        'meta_ppo_ant_nonstationary_learned_lowmem_l2reg',
    ]

    for exp in meta_experiments:
        if os.path.exists(f'./logs/{exp}'):
            check_metappo_training(exp)

    # Check test experiments
    print("\n\n2. TEST EXPERIMENTS (use learned optimizers)")
    print("-" * 60)

    test_experiments = [
        'ppo_ant_nonstationary_test_learned',
        'ppo_ant_nonstationary_test_learned_l2reg',
    ]

    for exp in test_experiments:
        if os.path.exists(f'./logs/{exp}'):
            check_test_results(exp)

    # Check baseline PPO
    print("\n\n3. BASELINE PPO (for comparison)")
    print("-" * 60)

    baseline_experiments = [
        'ppo_ant_nonstationary_adam',
    ]

    for exp in baseline_experiments:
        if os.path.exists(f'./logs/{exp}'):
            check_test_results(exp)

    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
MetaPPO Workflow:
1. Train MetaPPO → Learns optimizer parameters (param.pickle)
   - No return values in log (this is normal)
   - meta_ppo_* experiments

2. Test with learned optimizer → Measures actual performance
   - Return values in result_Test.feather
   - ppo_*_test_learned experiments (use Optim4RL with param_load_path)

3. Compare with baseline → Adam/RMSProp performance
   - ppo_* experiments with standard optimizers
    """)
