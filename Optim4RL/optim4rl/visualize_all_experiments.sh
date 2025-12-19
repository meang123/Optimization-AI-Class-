#!/bin/bash
# Quick Visualization Script for Available Experiments

cd '/home/maeng/바탕화면/optimization AI/Optim4RL/optim4rl'

echo "🎬 Visualizing Non-Stationary Ant Experiments..."
echo ""

# 1. Basic PPO (ppo_ant_non_tested)
echo "1/3 Visualizing Basic PPO (Adam optimizer)..."
python visualize_brax_ant.py \
  --policy_path './logs/ppo_ant_non_tested/1/inference_param_iter5.pickle' \
  --non_stationary \
  --num_episodes 3 \
  --output_dir './videos_comparison/basic_ppo'

# 2. Learned Optimizer
echo ""
echo "2/3 Visualizing Learned Optimizer..."
python visualize_brax_ant.py \
  --policy_path './logs/ppo_ant_nonstationary_test_learned/1/inference_param_iter5.pickle' \
  --non_stationary \
  --num_episodes 3 \
  --output_dir './videos_comparison/learned_optim'

# 3. Learned Optimizer + L2 Regularization
echo ""
echo "3/3 Visualizing Learned Optimizer + L2 Regularization..."
python visualize_brax_ant.py \
  --policy_path './logs/ppo_ant_nonstationary_test_learned_l2reg/1/inference_param_iter5.pickle' \
  --non_stationary \
  --num_episodes 3 \
  --output_dir './videos_comparison/learned_l2reg'

echo ""
echo "✅ 완료!"
echo ""
echo "📁 결과 위치:"
echo "   videos_comparison/basic_ppo/index.html"
echo "   videos_comparison/learned_optim/index.html"
echo "   videos_comparison/learned_l2reg/index.html"
echo ""
echo "🌐 브라우저에서 열기:"
echo "   firefox videos_comparison/basic_ppo/index.html"
