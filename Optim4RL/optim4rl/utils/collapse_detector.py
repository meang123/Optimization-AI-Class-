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
Policy Collapse Detection Utilities

This module provides tools for detecting and analyzing policy collapse in RL agents.
Policy collapse is characterized by sudden performance drops, loss of learned skills,
and inability to recover.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


def detect_collapse(
    episode_rewards: np.ndarray,
    window_size: int = 100,
    threshold: float = 0.5,
    min_baseline_length: int = 50
) -> List[Dict]:
    """
    Detect policy collapse events using rolling window performance drop detection.

    A collapse is detected when performance drops by more than threshold percentage
    compared to recent baseline performance.

    Args:
        episode_rewards: Array of episode returns over time
        window_size: Size of rolling window for baseline calculation
        threshold: Fraction of performance drop to consider as collapse (0.5 = 50% drop)
        min_baseline_length: Minimum number of episodes to establish baseline

    Returns:
        List of collapse events, each as dict with:
            - step: Episode index where collapse occurred
            - baseline: Average performance before collapse
            - collapsed_value: Performance at collapse
            - drop_percentage: Percentage drop from baseline
    """
    if len(episode_rewards) < min_baseline_length + window_size:
        return []

    collapse_events = []

    # Calculate rolling mean baseline
    baseline = pd.Series(episode_rewards).rolling(window=window_size, min_periods=min_baseline_length).mean()

    for i in range(min_baseline_length + window_size, len(episode_rewards)):
        current_baseline = baseline.iloc[i - 1]  # Use previous baseline
        current_value = episode_rewards[i]

        if pd.isna(current_baseline) or current_baseline <= 0:
            continue

        # Calculate drop percentage
        drop = (current_baseline - current_value) / abs(current_baseline)

        # Detect collapse
        if drop >= threshold:
            collapse_events.append({
                'step': i,
                'baseline': float(current_baseline),
                'collapsed_value': float(current_value),
                'drop_percentage': float(drop * 100)
            })

    return collapse_events


def compute_stability_score(episode_rewards: np.ndarray, window_size: int = 100) -> float:
    """
    Compute performance stability score (lower is more stable).

    Args:
        episode_rewards: Array of episode returns
        window_size: Window for computing rolling statistics

    Returns:
        Stability score (coefficient of variation averaged over rolling windows)
    """
    if len(episode_rewards) < window_size:
        return float('inf')

    rolling_mean = pd.Series(episode_rewards).rolling(window=window_size).mean()
    rolling_std = pd.Series(episode_rewards).rolling(window=window_size).std()

    # Coefficient of variation (CV) = std / mean
    cv = rolling_std / (rolling_mean + 1e-8)  # Add small epsilon to avoid division by zero

    # Return mean CV (excluding NaN from initial window)
    return float(cv.dropna().mean())


def compute_action_entropy(actions: np.ndarray, num_bins: int = 20) -> float:
    """
    Compute action entropy to detect policy collapse (low entropy = repetitive actions).

    Args:
        actions: Array of actions (shape: [timesteps, action_dim])
        num_bins: Number of bins for discretization

    Returns:
        Average entropy across action dimensions
    """
    if len(actions) == 0:
        return 0.0

    # Handle both 1D and 2D action arrays
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)

    entropies = []
    for dim in range(actions.shape[1]):
        # Discretize continuous actions into bins
        hist, _ = np.histogram(actions[:, dim], bins=num_bins, density=True)
        # Normalize to get probabilities
        probs = hist / (hist.sum() + 1e-8)
        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        entropies.append(entropy)

    return float(np.mean(entropies))


def analyze_collapse_recovery(
    episode_rewards: np.ndarray,
    collapse_events: List[Dict],
    recovery_window: int = 200,
    recovery_threshold: float = 0.8
) -> List[Dict]:
    """
    Analyze recovery patterns after collapse events.

    Args:
        episode_rewards: Array of episode returns
        collapse_events: List of collapse events from detect_collapse()
        recovery_window: Number of episodes to check for recovery
        recovery_threshold: Fraction of baseline to consider as recovered (0.8 = 80%)

    Returns:
        List of recovery analysis for each collapse, with:
            - collapse_step: Step where collapse occurred
            - recovered: Whether agent recovered
            - recovery_time: Episodes needed to recover (None if not recovered)
            - max_recovery: Maximum performance achieved in recovery window
    """
    recovery_analysis = []

    for event in collapse_events:
        collapse_step = event['step']
        baseline = event['baseline']
        recovery_target = baseline * recovery_threshold

        # Check recovery window
        window_end = min(collapse_step + recovery_window, len(episode_rewards))
        recovery_window_rewards = episode_rewards[collapse_step:window_end]

        # Find if and when recovery occurred
        recovered = False
        recovery_time = None
        max_recovery = float(recovery_window_rewards.max()) if len(recovery_window_rewards) > 0 else 0.0

        for t, reward in enumerate(recovery_window_rewards):
            if reward >= recovery_target:
                recovered = True
                recovery_time = t
                break

        recovery_analysis.append({
            'collapse_step': collapse_step,
            'recovered': recovered,
            'recovery_time': recovery_time,
            'max_recovery': max_recovery,
            'recovery_percentage': (max_recovery / baseline * 100) if baseline > 0 else 0.0
        })

    return recovery_analysis


def compare_collapse_rates(
    ppo_rewards: np.ndarray,
    metappo_rewards: np.ndarray,
    **detect_kwargs
) -> Dict[str, any]:
    """
    Compare policy collapse rates between PPO and MetaPPO.

    Args:
        ppo_rewards: Episode rewards from PPO training
        metappo_rewards: Episode rewards from MetaPPO training
        **detect_kwargs: Additional arguments for detect_collapse()

    Returns:
        Dictionary with comparative statistics:
            - ppo_collapse_count: Number of collapses in PPO
            - metappo_collapse_count: Number of collapses in MetaPPO
            - ppo_collapse_rate: Collapse rate (collapses per 1000 episodes)
            - metappo_collapse_rate: Collapse rate for MetaPPO
            - ppo_stability: Stability score for PPO
            - metappo_stability: Stability score for MetaPPO
    """
    ppo_collapses = detect_collapse(ppo_rewards, **detect_kwargs)
    metappo_collapses = detect_collapse(metappo_rewards, **detect_kwargs)

    ppo_stability = compute_stability_score(ppo_rewards)
    metappo_stability = compute_stability_score(metappo_rewards)

    return {
        'ppo_collapse_count': len(ppo_collapses),
        'metappo_collapse_count': len(metappo_collapses),
        'ppo_collapse_rate': len(ppo_collapses) / len(ppo_rewards) * 1000,
        'metappo_collapse_rate': len(metappo_collapses) / len(metappo_rewards) * 1000,
        'ppo_stability': ppo_stability,
        'metappo_stability': metappo_stability,
        'ppo_collapse_events': ppo_collapses,
        'metappo_collapse_events': metappo_collapses
    }


def generate_collapse_summary(collapse_events: List[Dict]) -> Dict[str, float]:
    """
    Generate summary statistics for collapse events.

    Args:
        collapse_events: List of collapse events from detect_collapse()

    Returns:
        Summary statistics:
            - total_collapses: Total number of collapse events
            - mean_drop: Average performance drop percentage
            - max_drop: Maximum performance drop percentage
            - min_baseline: Lowest baseline before collapse
            - max_baseline: Highest baseline before collapse
    """
    if not collapse_events:
        return {
            'total_collapses': 0,
            'mean_drop': 0.0,
            'max_drop': 0.0,
            'min_baseline': 0.0,
            'max_baseline': 0.0
        }

    drops = [event['drop_percentage'] for event in collapse_events]
    baselines = [event['baseline'] for event in collapse_events]

    return {
        'total_collapses': len(collapse_events),
        'mean_drop': float(np.mean(drops)),
        'max_drop': float(np.max(drops)),
        'min_baseline': float(np.min(baselines)),
        'max_baseline': float(np.max(baselines))
    }
