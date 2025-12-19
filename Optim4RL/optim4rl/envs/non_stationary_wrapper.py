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
Non-Stationary Dynamics Wrapper for Brax Environments

This wrapper modifies environment dynamics (NOT rewards) to create non-stationary
tasks for testing agent adaptability and policy collapse resistance.

Key features:
- Motor wear simulation (action scaling)
- Ground friction changes (velocity damping)
- Asymmetric fatigue (for humanoid)
- Intermittent faults
"""

import jax
import jax.numpy as jnp
from brax.envs import Wrapper


class NonStationaryDynamicsWrapper(Wrapper):
    """
    Wrapper that introduces non-stationary dynamics changes without modifying rewards.

    Args:
        env: Brax environment to wrap
        task_type: 'ant' or 'humanoid' - determines which dynamics modifications to apply
        change_mode: 'continuous' (gradual changes) or 'abrupt' (sudden changes)
    """

    def __init__(self, env, task_type='ant', change_mode='continuous'):
        super().__init__(env)
        self.task_type = task_type.lower()
        self.change_mode = change_mode

        # Validate task type
        if self.task_type not in ['ant', 'humanoid']:
            raise ValueError(f"task_type must be 'ant' or 'humanoid', got '{self.task_type}'")

    def reset(self, rng: jnp.ndarray):
        """Reset environment and initialize step counter."""
        state = self.env.reset(rng)
        # Add total_step counter to info dict for tracking dynamics changes
        state.info['total_step'] = jnp.array(0.0)
        return state

    def step(self, state, action):
        """
        Execute environment step with modified dynamics.

        Steps:
        1. Get current timestep
        2. Apply task-specific dynamics modifications to action
        3. Execute step with modified action
        4. Apply velocity damping to simulate ground friction changes
        5. Update step counter
        """
        # 1. Get current timestep
        t = state.info.get('total_step', jnp.array(0.0))

        # 2. Apply task-specific dynamics modifications
        if self.task_type == 'ant':
            modified_action, velocity_damp = self._modify_ant_dynamics(action, t)
        elif self.task_type == 'humanoid':
            modified_action, velocity_damp = self._modify_humanoid_dynamics(action, t)
        else:
            modified_action = action
            velocity_damp = 1.0

        # 3. Execute step with modified action
        next_state = self.env.step(state, modified_action)

        # 4. Apply velocity damping (simulates ground friction changes)
        # Note: pipeline_state contains qd (velocity) information
        # Always apply damping (if velocity_damp == 1.0, no effect)
        new_qd = next_state.pipeline_state.qd * velocity_damp
        new_pipeline_state = next_state.pipeline_state.replace(qd=new_qd)
        next_state = next_state.replace(pipeline_state=new_pipeline_state)

        # 5. Update step counter
        next_state.info['total_step'] = t + 1.0

        return next_state

    def _modify_ant_dynamics(self, action, t):
        """
        Modify Ant dynamics with motor wear and ground friction changes.

        Dynamics changes:
        1. Progressive motor wear: Output decreases over time
        2. Intermittent faults: Random action noise
        3. Ground friction: Periodic velocity damping changes

        Args:
            action: Original action from policy
            t: Current timestep

        Returns:
            modified_action: Action with wear/fault applied
            friction_damp: Velocity damping coefficient (0.9-1.0)
        """
        # A. Motor Wear (Progressive degradation)
        # Gradually reduce motor efficiency from 100% to 70%
        # Use slow sinusoidal variation to simulate wear cycles
        wear_factor = 1.0 - 0.3 * jnp.clip(jnp.sin(t / 100000.0), 0.0, 1.0)

        # B. Intermittent Fault (Random noise injection)
        # Add small random noise to simulate control instability
        # Use timestep as seed for reproducibility
        rng_key = jax.random.PRNGKey(jnp.int32(t))
        noise = jax.random.normal(rng_key, action.shape) * 0.05

        # Apply wear and noise
        modified_action = action * wear_factor + noise

        # C. Ground Friction (Velocity damping)
        # Simulate changing terrain: sticky (0.9) to normal (1.0)
        # Use slower cycle for ground changes
        friction_damp = 0.95 + 0.05 * jnp.cos(t / 50000.0)

        return modified_action, friction_damp

    def _modify_humanoid_dynamics(self, action, t):
        """
        Modify Humanoid dynamics with asymmetric fatigue and slippery ground.

        Dynamics changes:
        1. Asymmetric muscle fatigue: One side weakens progressively
        2. Slippery ground: Reduced action effectiveness

        Args:
            action: Original action from policy
            t: Current timestep

        Returns:
            modified_action: Action with asymmetric fatigue applied
            slippery_factor: Velocity modification (currently 1.0)
        """
        # A. Asymmetric Fatigue (Simulate muscle imbalance)
        # Divide action dimensions in half (left/right side approximation)
        half_dim = action.shape[-1] // 2

        # Left side progressively weakens over time
        # Starts at 100%, gradually reduces to 60%
        left_factor = 1.0 - 0.4 * jnp.clip(t / 1000000.0, 0.0, 1.0)
        left_factor = jnp.clip(left_factor, 0.6, 1.0)

        # Create mask: left side weakened, right side normal
        left_mask = jnp.full((half_dim,), left_factor)
        right_mask = jnp.ones((action.shape[-1] - half_dim,))
        fatigue_mask = jnp.concatenate([left_mask, right_mask])

        # Apply asymmetric fatigue
        modified_action = action * fatigue_mask

        # B. Slippery Ground
        # For humanoid, we focus on asymmetric fatigue
        # Velocity damping set to 1.0 (no change)
        slippery_factor = 1.0

        return modified_action, slippery_factor
