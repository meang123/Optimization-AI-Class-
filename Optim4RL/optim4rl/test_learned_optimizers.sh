#!/bin/bash
# Test script for learned optimizers

echo "=== Testing Learned Optimizers on Non-Stationary Ant ==="

# Test L2reg version
echo "Testing L2reg learned optimizer..."
python main.py --config_file ./configs/ppo_ant_nonstationary_test_learned_l2reg.json --config_idx 1

# Test baseline learned optimizer (without L2reg)
echo "Testing baseline learned optimizer..."
python main.py --config_file ./configs/ppo_ant_nonstationary_test_learned.json --config_idx 1

echo "=== Testing Complete ==="
echo "Results saved to logs/ppo_ant_nonstationary_test_learned*/1/result_Test.feather"
