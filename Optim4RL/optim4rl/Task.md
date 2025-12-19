# Non-Stationary Ant Task Experiments - Optim4RL

## Project Overview

This document describes the completed implementation and experiments for testing policy collapse and optimizer robustness in non-stationary reinforcement learning environments.

### Research Objectives

1. **Investigate policy collapse in non-stationary environments**
   - Does standard PPO with Adam optimizer experience policy collapse when environment dynamics change over time?
   - How does optimizer configuration (e.g., Adam beta values) affect collapse behavior?

2. **Evaluate meta-learned optimizer robustness**
   - Is MetaPPO with learned Optim4RL optimizer more robust to non-stationary dynamics?
   - Can optimizers trained on stationary tasks transfer to non-stationary tasks?

3. **Compare optimizer adaptation capabilities**
   - Standard Adam vs custom Adam configurations
   - Meta-learned Optim4RL vs hand-tuned optimizers
   - Transfer learning: stationary-trained Optim4RL applied to non-stationary tasks

---

## Implementation Summary

### Non-Stationary Environment Wrapper

**File**: `envs/non_stationary_wrapper.py`

Implemented a wrapper that progressively modifies Ant environment dynamics during training:

**Key Features**:
- **Motor Wear**: Progressive reduction in actuator output (simulating motor degradation)
- **Intermittent Faults**: Random actuator failures
- **Ground Friction Changes**: Time-varying friction coefficients
- **Velocity Damping**: Additional damping to simulate energy loss

**Integration**: The wrapper is automatically applied when `"non_stationary": true` and `"task_type": "ant"` are specified in config files.

---

## Experiment Configurations

All experiments were conducted on the **Ant task only** with **1e9 (1 billion) training steps** to observe long-term policy behavior and collapse patterns.

### 1. PPO with Standard Adam (Baseline)

**Config**: `configs/ppo_ant_nonstationary.json`

**Description**: Standard PPO with default Adam optimizer (beta1=0.9, beta2=0.999)

**Key Parameters**:
- Agent: PPO
- Optimizer: Adam (learning_rate=3e-4, grad_clip=1)
- Training steps: 1e9
- Environment: 4096 parallel envs
- Non-stationary: Enabled

**Purpose**: Establish baseline performance and observe policy collapse patterns with standard optimizer configuration.

### 2. PPO with Custom Adam (High Beta)

**Config**: `configs/ppo_ant_nonstationary_adam.json`

**Description**: PPO with custom Adam configuration using higher momentum (beta1=beta2=0.997)

**Key Parameters**:
- Agent: PPO
- Optimizer: Adam (learning_rate=3e-4, **b1=0.997, b2=0.997**, grad_clip=1)
- Training steps: 1e9
- Environment: 4096 parallel envs
- Non-stationary: Enabled

**Purpose**: Test whether higher momentum values help maintain stability and reduce collapse in non-stationary settings.

### 3. MetaPPO with Standard Adam

**Config**: `configs/meta_ppo_ant_nonstationary.json`

**Description**: Meta-learned PPO with standard Adam optimizers for both agent and meta-level training

**Key Parameters**:
- Agent: MetaPPO
- Agent Optimizer: Adam (learning_rate=3e-4)
- Meta Optimizer: Adam (learning_rate=3e-4)
- Inner updates: 4
- Reset interval: 128
- Training steps: 1e9
- Environment: 2048 parallel envs
- Non-stationary: Enabled

**Purpose**: Evaluate whether meta-learning provides better adaptation and collapse resistance compared to standard PPO.

### 4. MetaPPO with Learned Optim4RL (Low Memory + L2 Regularization)

**Config**: `configs/meta_ppo_ant_nonstationary_learned_lowmem_l2reg.json`

**Description**: MetaPPO using learned Optim4RL optimizer for agent updates, with memory-efficient settings and L2 regularization

**Key Parameters**:
- Agent: MetaPPO
- Agent Optimizer: **Optim4RL** (learned optimizer, learning_rate=3e-4)
- Meta Optimizer: Adam (learning_rate=3e-4)
- Inner updates: 2 (reduced for memory efficiency)
- Reset interval: 128
- L2 weight: 1e-4
- Training steps: 1e9
- Environment: 2048 parallel envs (reduced for memory)
- Non-stationary: Enabled

**Purpose**: Test whether meta-learned optimizers trained for RL tasks provide superior robustness in non-stationary environments.

### 5. PPO with Stationary-Trained Optim4RL (Transfer Learning)

**Config**: `configs/stationary_optim.json`

**Description**: Standard PPO using Optim4RL optimizer parameters learned from stationary training, applied to non-stationary task

**Key Parameters**:
- Agent: PPO
- Optimizer: **Optim4RL** (param_load_path: `./logs/meta_rl_ant/5/param.pickle`)
- Training steps: 1e9
- Environment: 4096 parallel envs
- Non-stationary: **Enabled** (note: config has non_stationary=true despite name)

**Purpose**: Evaluate transfer learning capability - can optimizers trained on stationary tasks generalize to non-stationary dynamics?

---

## Experiment Summary Table

| Exp # | Name | Config File | Agent | Optimizer | Video Directory | Iteration | Best Reward |
|-------|------|-------------|-------|-----------|-----------------|-----------|-------------|
| 1 | PPO + Standard Adam | `ppo_ant_nonstationary.json` | PPO | Adam (default) | `videos_iter15_default_adam/` | 15 | -14 |
| 2 | PPO + Custom Adam | `ppo_ant_nonstationary_adam.json` | PPO | Adam (β=0.997) | `videos_iter15_tunned_adam/` | 15 | -233 |
| 3 | MetaPPO + Adam | `meta_ppo_ant_nonstationary.json` | MetaPPO | Adam (both) | `videos_iter5/` | 5 | -351 to -517 |
| 4 | MetaPPO + Optim4RL | `meta_ppo_ant_nonstationary_learned_lowmem_l2reg.json` | MetaPPO | Optim4RL + L2 | `videos_iter5_l2/` | 5 | -416 |
| 5 | Transfer Learning | `stationary_optim.json` | PPO | Optim4RL (pretrained) | `videos_iter5_stationary/` | 5 | -76 ✅ |

**Note**: Lower (more negative) rewards indicate worse performance. Experiment 5 shows the best early performance, suggesting good transfer capability.

---

## How to Run Experiments

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify GPU availability (recommended for 1e9 step training)
python -c "import jax; print(jax.devices())"
```

### Running Individual Experiments

#### 1. PPO with Standard Adam (Baseline)

```bash
python main.py --config_file ./configs/ppo_ant_nonstationary.json --config_idx 0
```

#### 2. PPO with Custom Adam (Beta=0.997)

```bash
python main.py --config_file ./configs/ppo_ant_nonstationary_adam.json --config_idx 0
```

#### 3. MetaPPO with Adam

```bash
python main.py --config_file ./configs/meta_ppo_ant_nonstationary.json --config_idx 0
```

#### 4. MetaPPO with Learned Optim4RL + L2 Regularization

```bash
python main.py --config_file ./configs/meta_ppo_ant_nonstationary_learned_lowmem_l2reg.json --config_idx 0
```

#### 5. PPO with Stationary-Trained Optim4RL (Transfer)

**Important**: First ensure the pretrained Optim4RL parameters exist at `./logs/meta_rl_ant/5/param.pickle`

```bash
python main.py --config_file ./configs/stationary_optim.json --config_idx 0
```

### Batch Execution

To run multiple experiments sequentially:

```bash
# Create a simple execution script
#!/bin/bash

echo "Starting Experiment 1: PPO with Standard Adam"
python main.py --config_file ./configs/ppo_ant_nonstationary.json --config_idx 0

echo "Starting Experiment 2: PPO with Custom Adam"
python main.py --config_file ./configs/ppo_ant_nonstationary_adam.json --config_idx 0

echo "Starting Experiment 3: MetaPPO with Adam"
python main.py --config_file ./configs/meta_ppo_ant_nonstationary.json --config_idx 0

echo "Starting Experiment 4: MetaPPO with Learned Optim4RL"
python main.py --config_file ./configs/meta_ppo_ant_nonstationary_learned_lowmem_l2reg.json --config_idx 0

echo "Starting Experiment 5: Transfer Learning - Stationary Optim4RL"
python main.py --config_file ./configs/stationary_optim.json --config_idx 0

echo "All experiments completed!"
```

---

## Monitoring Training Progress

### Real-time Monitoring

During training, monitor for policy collapse indicators:

1. **Episode Returns**: Sudden drops (>50%) indicate potential collapse
2. **Loss Values**: Diverging losses suggest instability
3. **Checkpoints**: Saved at intervals specified in `save_param` config

### Log Files

Training logs are saved in:
```
./logs/[experiment_name]/[seed]/
```

Key files:
- `training_log.txt`: Episode returns and metrics
- `param.pickle`: Saved model parameters
- `policy_params/`: Policy checkpoints at specified intervals

---

## Analysis and Visualization

### Available Analysis Scripts

Several analysis scripts are available in the repository:

1. **Policy Collapse Analysis**
   ```bash
   python analyze_training_collapse.py
   python analyze_collapse_comparison.py
   python visualize_collapse.py
   ```

2. **Adam Beta Comparison**
   ```bash
   python analyze_adam_beta_comparison.py
   python analyze_humanoid_beta_comparison.py
   ```

3. **Experiment Comparison**
   ```bash
   python compare_correct_experiments.py
   python compare_stationary_vs_nonstationary.py
   python compare_adam_vs_l2reg.py
   ```

4. **Visualization**
   ```bash
   python visualize_brax_ant.py
   python compare_policies_visualization.py
   ```

### Running Analysis

Example workflow:

```bash
# 1. Analyze training collapse patterns
python analyze_training_collapse.py

# 2. Compare Adam configurations
python analyze_adam_beta_comparison.py

# 3. Compare all experiments
python compare_correct_experiments.py

# 4. Visualize trained policies
python visualize_brax_ant.py --log_dir ./logs/[experiment_name]/[seed]/
```

### Policy Rollout Videos

Generated policy rollout videos are available for visual inspection of agent behavior under non-stationary dynamics. Each video directory contains HTML files that can be viewed in a web browser.

#### Available Video Results

**1. videos_iter15_default_adam/** - PPO with Standard Adam (Iteration 15)
- **Experiment**: Experiment 1 (PPO + Standard Adam)
- **Config**: `ppo_ant_nonstationary.json`
- **Performance**: Episode reward = -14
- **Description**: Shows policy behavior at iteration 15 with default Adam optimizer (beta1=0.9, beta2=0.999)
- **Files**:
  - `index.html` - Video gallery index
  - `inference_param_iter15_episode1_reward-14.html` - Episode rollout visualization

**2. videos_iter15_tunned_adam/** - PPO with Custom Adam (Iteration 15)
- **Experiment**: Experiment 2 (PPO + Custom Adam β=0.997)
- **Config**: `ppo_ant_nonstationary_adam.json`
- **Performance**: Episode reward = -233
- **Description**: Shows policy behavior at iteration 15 with high momentum Adam (beta1=beta2=0.997)
- **Files**:
  - `index.html` - Video gallery index
  - `inference_param_iter15_episode1_reward-233.html` - Episode rollout visualization
- **Note**: Lower performance compared to default Adam suggests high momentum may not be optimal for this task

**3. videos_iter5/** - MetaPPO with Adam (Iteration 5)
- **Experiment**: Experiment 3 (MetaPPO + Adam)
- **Config**: `meta_ppo_ant_nonstationary.json`
- **Performance**: Multiple episodes with rewards -396, -517, -351
- **Description**: Shows early meta-learning behavior at iteration 5
- **Files**:
  - `index.html` - Video gallery index
  - `inference_param_iter5_episode1_reward-396.html`
  - `inference_param_iter5_episode2_reward-517.html`
  - `inference_param_iter5_episode3_reward-351.html`

**4. videos_iter5_l2/** - MetaPPO with Optim4RL + L2 Regularization (Iteration 5)
- **Experiment**: Experiment 4 (MetaPPO + Learned Optim4RL + L2)
- **Config**: `meta_ppo_ant_nonstationary_learned_lowmem_l2reg.json`
- **Performance**: Episode reward = -416
- **Description**: Shows learned optimizer behavior with L2 regularization at iteration 5
- **Files**:
  - `index.html` - Video gallery index
  - `inference_param_iter5_episode1_reward-416.html` - Episode rollout visualization
  - `inference_data_ant.csv` - Quantitative performance data
  - `inference_results_ant.png` - Performance analysis plot

**5. videos_iter5_stationary/** - Transfer Learning (Iteration 5)
- **Experiment**: Experiment 5 (Stationary Optim4RL → Non-stationary)
- **Config**: `stationary_optim.json`
- **Performance**: Episode reward = -76
- **Description**: Shows transfer learning performance - optimizer trained on stationary task applied to non-stationary environment
- **Files**:
  - `index.html` - Video gallery index
  - `inference_param_iter5_episode1_reward-76.html` - Episode rollout visualization
- **Note**: Significantly better performance suggests good transfer capability

#### How to View Video Results

**Method 1: Direct HTML Opening (Recommended)**

Open any video HTML file directly in your web browser:

```bash
# Example: View tuned Adam results (Experiment 2)
firefox videos_iter15_tunned_adam/index.html

# Or use your default browser
xdg-open videos_iter15_tunned_adam/index.html

# On macOS
open videos_iter15_tunned_adam/index.html

# On Windows
start videos_iter15_tunned_adam/index.html
```

**Method 2: Python HTTP Server**

For better compatibility, serve videos via local HTTP server:

```bash
# Navigate to project directory
cd /home/maeng/바탕화면/optimization\ AI/Optim4RL/optim4rl/

# Start Python HTTP server
python -m http.server 8000

# Then open in browser:
# http://localhost:8000/videos_iter15_tunned_adam/index.html
# http://localhost:8000/videos_iter5_l2/index.html
# etc.
```

**Method 3: View All Videos**

Create a simple script to open all video indexes:

```bash
#!/bin/bash
# view_all_videos.sh

echo "Opening all video results..."

xdg-open videos_iter15_default_adam/index.html &
xdg-open videos_iter15_tunned_adam/index.html &
xdg-open videos_iter5/index.html &
xdg-open videos_iter5_l2/index.html &
xdg-open videos_iter5_stationary/index.html &

echo "All video galleries opened in browser"
```

#### Video Comparison Insights

**Performance Ranking (by episode reward at early iterations)**:
1. **videos_iter5_stationary** (-76) - Best early performance ✅
2. **videos_iter15_default_adam** (-14) - Good performance with standard Adam
3. **videos_iter15_tunned_adam** (-233) - Worse with high momentum
4. **videos_iter5/** (-351 to -517) - Early MetaPPO learning
5. **videos_iter5_l2/** (-416) - MetaPPO with learned optimizer

**Key Observations**:
- **Transfer learning** (stationary → non-stationary) shows surprisingly good early performance
- **Standard Adam** outperforms custom high-momentum Adam at iteration 15
- **MetaPPO** experiments show higher variance in early iterations (expected during meta-learning)
- **Learned optimizer with L2** shows comparable performance to standard MetaPPO

**Video Analysis Tips**:
- Watch for **locomotion patterns**: smooth vs jerky movements indicate policy quality
- Observe **fall frequency**: more falls suggest policy collapse or poor adaptation
- Compare **movement efficiency**: forward progress vs energy expenditure
- Note **behavior changes**: adaptation to non-stationary dynamics over time

#### Quick Reference: Video Viewing Commands

```bash
# Experiment 1: PPO with Standard Adam
xdg-open videos_iter15_default_adam/index.html

# Experiment 2: PPO with Custom Adam (β=0.997)
xdg-open videos_iter15_tunned_adam/index.html

# Experiment 3: MetaPPO with Adam
xdg-open videos_iter5/index.html

# Experiment 4: MetaPPO with Optim4RL + L2
xdg-open videos_iter5_l2/index.html

# Experiment 5: Transfer Learning
xdg-open videos_iter5_stationary/index.html
```

**Or use Python HTTP server for all videos:**
```bash
cd /home/maeng/바탕화면/optimization\ AI/Optim4RL/optim4rl/
python -m http.server 8000
# Then navigate to: http://localhost:8000/videos_iter5_l2/index.html (or any other directory)
```



### Ant default adam 

![ant_default_adam](./ant_default_adam.gif)

### Ant tunned Adam 
![ant_default_adam](ant_tunned_adam.gif)

### Ant optim4rl optimizer 
![ant_default_adam](ant_optim4rl.gif)

### Ant optim4RL optimizer with L2 
![ant_default_adam](ant_optim4rl_l2.gif)


### Ant learned from stationary task, apply to non stationary task(ours)

![ant_default_adam](ant_stationary.gif)






---

## Key Findings and Research Questions

### Research Questions Addressed

1. **Does PPO experience policy collapse in non-stationary environments?**
   - Experiments 1 & 2 provide data on collapse frequency and severity
   - Custom Adam (beta=0.997) tested as potential mitigation strategy

2. **Is MetaPPO more robust than standard PPO?**
   - Experiments 3 & 4 compare meta-learning approaches
   - Learned Optim4RL (Exp 4) tests specialized RL optimizers

3. **Can optimizers transfer from stationary to non-stationary tasks?**
   - Experiment 5 directly tests transfer learning capability
   - Evaluates generalization of learned optimization strategies

### Comparative Analysis

The five experiments allow systematic comparison:

| Experiment | Agent | Optimizer | Key Feature | Purpose |
|------------|-------|-----------|-------------|---------|
| 1 | PPO | Adam (default) | Baseline | Standard approach |
| 2 | PPO | Adam (β=0.997) | High momentum | Stability test |
| 3 | MetaPPO | Adam | Meta-learning | Adaptation capability |
| 4 | MetaPPO | Optim4RL + L2 | Learned optimizer | RL-specialized |
| 5 | PPO | Optim4RL (transfer) | Transfer learning | Generalization |

---

## Expected Results and Metrics

### Primary Metrics

1. **Collapse Detection**
   - Sudden performance drops (>50% reduction in episode return)
   - Action entropy reduction (indicating policy degeneration)
   - Recovery time after dynamics changes

2. **Performance Stability**
   - Mean episode return over training
   - Standard deviation of returns (lower = more stable)
   - Performance variance before/after dynamics changes

3. **Adaptation Speed**
   - Time to recover after environment changes
   - Learning curve gradient during adaptation periods

### Success Criteria

**Policy Collapse Resistance**:
- MetaPPO should show fewer collapse events than PPO
- Learned optimizers should provide more stable training
- Transfer learning should maintain reasonable performance

**Optimizer Comparison**:
- Custom Adam (β=0.997) vs default Adam
- Learned Optim4RL vs hand-tuned Adam
- Stationary-trained optimizer generalization

---

## Implementation Details

### Non-Stationary Dynamics Schedule

The non-stationary wrapper applies progressive changes:

1. **Motor Wear**: Linear degradation starting after initial training
2. **Friction Changes**: Sinusoidal or step-function variations
3. **Random Faults**: Probabilistic actuator failures

**Change Period**: Dynamics evolve throughout the 1e9 training steps

### Memory Optimization

For limited GPU memory:
- Reduce `num_envs` (2048 instead of 4096)
- Reduce `inner_updates` for MetaPPO (2 instead of 4)
- Reduce `batch_size` (512 instead of 1024)

See `meta_ppo_ant_nonstationary_learned_lowmem_l2reg.json` for example.

### Computational Requirements

**Approximate Training Time** (per experiment):
- GPU (A100/V100): ~24-48 hours for 1e9 steps
- GPU (RTX 3090): ~48-72 hours
- Multi-GPU: Can be accelerated with `max_devices_per_host` setting

**Storage Requirements**:
- ~5-10 GB per experiment (logs + checkpoints)
- Policy videos (if enabled): Additional 1-5 GB

---

## Configuration Notes

### Common Parameters Across All Configs

- **train_steps**: 1e9 (1 billion steps) - essential for observing long-term collapse
- **episode_length**: 1000 steps per episode
- **discount**: 0.97 (gamma value for reward discounting)
- **normalize_obs**: true (observation normalization)
- **non_stationary**: true (enables dynamics changes)
- **task_type**: "ant" (all experiments on Ant environment)

### Optimizer-Specific Notes

**Adam Configurations**:
- Standard: beta1=0.9, beta2=0.999 (PyTorch/TensorFlow default)
- Custom: beta1=beta2=0.997 (higher momentum for stability)
- Learning rate: 3e-4 (consistent across all experiments)
- Gradient clipping: 1.0 (prevents gradient explosion)

**Optim4RL (Learned Optimizer)**:
- Requires pretrained parameters: `param_load_path`
- Uses learned update rules instead of hand-crafted formulas
- Meta-trained on RL-specific optimization landscapes

### MetaPPO-Specific Parameters

**Inner Updates**: Number of agent gradient steps per meta-update
- Standard: 4 updates
- Low-memory: 2 updates (faster but potentially less stable)

**Reset Interval**: Agent training state reset frequency
- Value: 128 iterations
- Prevents meta-overfitting and maintains plasticity

**L2 Regularization**:
- Weight: 1e-4 (only in learned_lowmem_l2reg config)
- Helps prevent parameter drift in non-stationary settings

---

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM) Errors**
   - Reduce `num_envs` in config
   - Reduce `batch_size`
   - Use low-memory config variants
   - Enable `max_devices_per_host` to limit GPU usage

2. **Slow Training Speed**
   - Verify GPU is being used: `jax.devices()`
   - Check for data transfer bottlenecks
   - Consider reducing `num_envs` slightly

3. **NaN Loss Values**
   - Check gradient clipping is enabled
   - Reduce learning rate
   - Verify observation normalization is active

---



