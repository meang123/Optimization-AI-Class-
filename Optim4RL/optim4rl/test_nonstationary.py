#!/usr/bin/env python3
"""
Quick test script for non-stationary experiments
Tests if PPO and MetaPPO can run with non-stationary wrapper
"""

import subprocess
import sys
import time
from pathlib import Path

def test_experiment(config_file, exp_name, timeout=60):
    """Test if experiment can start successfully."""
    print(f"\n{'='*80}")
    print(f"Testing: {exp_name}")
    print(f"Config: {config_file}")
    print(f"{'='*80}\n")

    cmd = f"timeout {timeout} python main.py --config_file {config_file} --config_idx 1"

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd="/home/maeng/바탕화면/optimization AI/Optim4RL/optim4rl"
        )

        # Check if log file was created
        log_path = Path(f"logs/{exp_name}/1/log.txt")
        if log_path.exists():
            print(f"✅ SUCCESS: Log file created")
            with open(log_path, 'r') as f:
                log_content = f.read()
                print(f"\nLog content:")
                print(log_content)
            return True
        else:
            print(f"❌ FAILED: Log file not created")
            print(f"STDERR:\n{result.stderr}")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    finally:
        elapsed = time.time() - start_time
        print(f"\nElapsed time: {elapsed:.1f}s")

def main():
    print("\n" + "="*80)
    print("NON-STATIONARY TASK TEST SUITE")
    print("="*80)

    experiments = [
        ("./configs/ppo_ant_nonstationary.json", "ppo_ant_nonstationary"),
        ("./configs/metap_ppo_ant_nonstationary.json", "metap_ppo_ant_nonstationary"),
    ]

    results = {}
    for config, name in experiments:
        results[name] = test_experiment(config, name, timeout=90)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    print()

    # Return exit code
    all_passed = all(results.values())
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
