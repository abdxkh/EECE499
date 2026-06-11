from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta, Categorical
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """
    Shared actor-critic MLP for PPO.

    The actor has two categorical heads per UAV:
    - sensor choice: num_sensors real sensors plus one idle action
    - power choice: num_power_levels transmit-power choices
    """

    def __init__(
        self,
        obs_dim,
        num_uavs,
        num_sensors,
        num_power_levels,
        hidden_dim=128,
        continuous_power=False,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.num_uavs = num_uavs
        self.num_sensors = num_sensors
        self.num_sensor_actions = num_sensors + 1
        self.num_power_levels = num_power_levels
        self.continuous_power = continuous_power

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.sensor_head = nn.Linear(hidden_dim, num_uavs * self.num_sensor_actions)
        if self.continuous_power:
            self.power_head = nn.Linear(hidden_dim, num_uavs * 2)
        else:
            self.power_head = nn.Linear(hidden_dim, num_uavs * num_power_levels)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        """
        Args:
            obs: tensor of shape (batch, obs_dim)

        Returns:
            sensor_logits: shape (batch, num_uavs, num_sensors + 1)
            power output:
                discrete mode: shape (batch, num_uavs, num_power_levels)
                continuous mode: Beta alpha/beta, shape (batch, num_uavs, 2)
            value: shape (batch,)
        """

        features = self.backbone(obs)

        sensor_logits = self.sensor_head(features)
        sensor_logits = sensor_logits.view(-1, self.num_uavs, self.num_sensor_actions)

        power_output = self.power_head(features)
        if self.continuous_power:
            power_output = power_output.view(-1, self.num_uavs, 2)
            power_output = F.softplus(power_output) + 1.0
        else:
            power_output = power_output.view(-1, self.num_uavs, self.num_power_levels)

        value = self.value_head(features).squeeze(-1)

        return sensor_logits, power_output, value


@dataclass
class RolloutMemory:
    """
    Simple rollout storage for PPO.
    """

    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def add(self, obs, action_indices, log_prob, reward, done, value):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(np.asarray(action_indices, dtype=np.float32))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.rewards)


class PPOAgent:
    """
    Reusable PPO learner.

    This class is intentionally environment-agnostic. It knows how to sample
    categorical actions from numeric observations and update an actor-critic
    network using PPO; it does not know AoDT, channels, buffers, or queues.
    """

    def __init__(
        self,
        obs_dim,
        num_uavs,
        num_sensors,
        num_power_levels,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=4,
        batch_size=64,
        hidden_dim=128,
        mask_worker_actions=False,
        worker_freshness_bias=0.0,
        force_max_power=False,
        continuous_power=False,
        power_min=None,
        power_max=None,
        device=None,
    ):
        self.obs_dim = obs_dim
        self.num_uavs = num_uavs
        self.num_sensors = num_sensors
        self.num_sensor_actions = num_sensors + 1
        self.num_power_levels = num_power_levels

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.mask_worker_actions = mask_worker_actions
        self.worker_freshness_bias = worker_freshness_bias
        self.force_max_power = force_max_power
        self.continuous_power = continuous_power
        self.power_min = (
            float(power_min)
            if power_min is not None
            else 0.0
        )
        self.power_max = (
            float(power_max)
            if power_max is not None
            else float(max(num_power_levels - 1, 1))
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(
            obs_dim=obs_dim,
            num_uavs=num_uavs,
            num_sensors=num_sensors,
            num_power_levels=num_power_levels,
            hidden_dim=hidden_dim,
            continuous_power=continuous_power,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, obs, deterministic=False):
        """
        Sample a worker action from the current policy.

        Returns:
            env_action:
                list of (sensor_id, power_index), one tuple per UAV.
                sensor_id is -1 for idle.
            action_indices:
                numpy array of shape (num_uavs, 2), using internal categorical
                sensor indices where num_sensors means idle.
            log_prob:
                summed log probability of all per-UAV categorical choices.
            value:
                critic value estimate for obs.
        """

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_tensor = obs_tensor.view(1, -1)

        with torch.no_grad():
            sensor_logits, power_output, value = self.model(obs_tensor)
            sensor_logits = self._apply_worker_freshness_bias(sensor_logits, obs_tensor)

            sensor_actions, sensor_log_prob, sensor_entropy = self._sample_sensor_actions(
                sensor_logits[0],
                obs_tensor[0],
                deterministic=deterministic,
            )

            if self.continuous_power:
                if self.force_max_power:
                    power_actions = torch.ones(
                        self.num_uavs,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    log_prob = sensor_log_prob
                else:
                    power_dist = Beta(
                        power_output[0, :, 0],
                        power_output[0, :, 1],
                    )
                    if deterministic:
                        power_actions = (
                            power_output[0, :, 0]
                            / (power_output[0, :, 0] + power_output[0, :, 1])
                        )
                    else:
                        power_actions = power_dist.sample()
                    log_prob = sensor_log_prob + power_dist.log_prob(power_actions).sum()
            elif self.force_max_power:
                power_actions = torch.full(
                    (self.num_uavs,),
                    self.num_power_levels - 1,
                    dtype=torch.float32,
                    device=self.device,
                )
                log_prob = sensor_log_prob
            else:
                power_dist = Categorical(logits=power_output[0])
                if deterministic:
                    power_actions = torch.argmax(power_output[0], dim=-1)
                else:
                    power_actions = power_dist.sample()
                log_prob = sensor_log_prob + power_dist.log_prob(power_actions).sum()

        env_action = self._format_env_action(sensor_actions, power_actions)
        action_indices = torch.stack([sensor_actions.float(), power_actions.float()], dim=1)

        return (
            env_action,
            action_indices.cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def compute_returns_and_advantages(
        self,
        rewards,
        dones,
        values,
        last_value=0.0,
    ):
        """
        Compute discounted returns and GAE advantages.
        """

        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae

        returns = advantages + values

        return returns.astype(np.float32), advantages.astype(np.float32)

    def update(self, memory, last_value=0.0):
        """
        Apply PPO updates using one rollout.

        Args:
            memory: RolloutMemory
            last_value: bootstrap critic value after the rollout, or 0 if done

        Returns:
            dictionary of final epoch loss scalars
        """

        if len(memory) == 0:
            return {}

        returns, advantages = self.compute_returns_and_advantages(
            memory.rewards,
            memory.dones,
            memory.values,
            last_value=last_value,
        )

        obs = torch.as_tensor(np.asarray(memory.obs), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(memory.actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(memory.log_probs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        num_steps = obs.shape[0]
        final_stats = {}

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(num_steps, device=self.device)

            for start in range(0, num_steps, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                new_log_probs, entropy, values = self._evaluate_actions(
                    obs[batch_idx],
                    actions[batch_idx],
                )

                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])

                unclipped = ratio * advantages[batch_idx]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                ) * advantages[batch_idx]

                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = (returns[batch_idx] - values).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = actor_loss + self.value_coef * critic_loss
                loss = loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                final_stats = {
                    "loss": float(loss.item()),
                    "actor_loss": float(actor_loss.item()),
                    "critic_loss": float(critic_loss.item()),
                    "entropy": float(entropy_loss.item()),
                }

        return final_stats

    def _evaluate_actions(self, obs, actions):
        """
        Evaluate log probabilities, entropy, and values for stored actions.
        """

        sensor_logits, power_output, values = self.model(obs)
        sensor_logits = self._apply_worker_freshness_bias(sensor_logits, obs)

        sensor_actions = actions[:, :, 0].long()
        power_actions = actions[:, :, 1]

        if self.mask_worker_actions:
            sensor_log_probs, sensor_entropy = self._evaluate_masked_sensor_actions(
                sensor_logits,
                obs,
                sensor_actions,
            )
        else:
            sensor_dist = Categorical(logits=sensor_logits)
            sensor_log_probs = sensor_dist.log_prob(sensor_actions).sum(dim=1)
            sensor_entropy = sensor_dist.entropy().sum(dim=1)

        if self.continuous_power:
            if self.force_max_power:
                log_probs = sensor_log_probs
                entropy = sensor_entropy
            else:
                power_dist = Beta(power_output[:, :, 0], power_output[:, :, 1])
                power_actions = torch.clamp(power_actions, 1e-6, 1.0 - 1e-6)
                log_probs = sensor_log_probs + power_dist.log_prob(power_actions).sum(dim=1)
                entropy = sensor_entropy + power_dist.entropy().sum(dim=1)
        elif self.force_max_power:
            log_probs = sensor_log_probs
            entropy = sensor_entropy
        else:
            power_dist = Categorical(logits=power_output)
            log_probs = sensor_log_probs + power_dist.log_prob(power_actions.long()).sum(dim=1)
            entropy = sensor_entropy + power_dist.entropy().sum(dim=1)

        return log_probs, entropy, values

    def _apply_worker_freshness_bias(self, sensor_logits, obs):
        """
        Add a soft prior toward sensors with higher current AoI.

        This is enabled only for the worker action layout. It does not mask or
        force a choice; PPO can still learn around the prior.
        """

        if self.worker_freshness_bias <= 0.0:
            return sensor_logits

        sensor_aoi_start = 3 * self.num_sensors
        sensor_aoi_end = sensor_aoi_start + self.num_sensors
        sensor_aoi = obs[:, sensor_aoi_start:sensor_aoi_end]

        biased_logits = sensor_logits.clone()
        biased_logits[:, :, :self.num_sensors] += (
            self.worker_freshness_bias * sensor_aoi.unsqueeze(1)
        )

        return biased_logits

    def _sample_sensor_actions(self, sensor_logits, obs, deterministic=False):
        """
        Sample worker sensor actions, optionally masking empty, duplicate, and
        unnecessary idle choices.
        """

        if not self.mask_worker_actions:
            sensor_dist = Categorical(logits=sensor_logits)
            if deterministic:
                sensor_actions = torch.argmax(sensor_logits, dim=-1)
            else:
                sensor_actions = sensor_dist.sample()
            return (
                sensor_actions,
                sensor_dist.log_prob(sensor_actions).sum(),
                sensor_dist.entropy().sum(),
            )

        selected = torch.zeros(self.num_sensors, dtype=torch.bool, device=self.device)
        sensor_actions = []
        log_probs = []
        entropies = []

        for m in range(self.num_uavs):
            mask = self._worker_sensor_mask(obs, selected)
            masked_logits = sensor_logits[m].masked_fill(~mask, -1e9)
            sensor_dist = Categorical(logits=masked_logits)
            if deterministic:
                action = torch.argmax(masked_logits)
            else:
                action = sensor_dist.sample()

            sensor_actions.append(action)
            log_probs.append(sensor_dist.log_prob(action))
            entropies.append(sensor_dist.entropy())

            action_int = int(action.item())
            if action_int < self.num_sensors:
                selected[action_int] = True

        return (
            torch.stack(sensor_actions),
            torch.stack(log_probs).sum(),
            torch.stack(entropies).sum(),
        )

    def _evaluate_masked_sensor_actions(self, sensor_logits, obs, sensor_actions):
        """
        Recompute masked sensor log probabilities for PPO updates.
        """

        batch_size = obs.shape[0]
        selected = torch.zeros(
            (batch_size, self.num_sensors),
            dtype=torch.bool,
            device=self.device,
        )
        log_probs = []
        entropies = []

        for m in range(self.num_uavs):
            mask = self._worker_sensor_mask_batch(obs, selected)
            masked_logits = sensor_logits[:, m, :].masked_fill(~mask, -1e9)
            sensor_dist = Categorical(logits=masked_logits)
            action = sensor_actions[:, m]

            log_probs.append(sensor_dist.log_prob(action))
            entropies.append(sensor_dist.entropy())

            real_sensor = action < self.num_sensors
            if real_sensor.any():
                rows = torch.arange(batch_size, device=self.device)[real_sensor]
                cols = action[real_sensor]
                selected[rows, cols] = True

        return torch.stack(log_probs, dim=1).sum(dim=1), torch.stack(entropies, dim=1).sum(dim=1)

    def _worker_sensor_mask(self, obs, selected):
        """
        Build a one-step worker sensor mask for one observation.

        The observation starts with Q, so obs[:num_sensors] gives pending flags.
        Idle is allowed only when no valid unselected pending sensor remains.
        """

        pending = obs[:self.num_sensors] > 0.5
        valid_sensors = pending & ~selected

        mask = torch.zeros(self.num_sensor_actions, dtype=torch.bool, device=self.device)
        mask[:self.num_sensors] = valid_sensors
        mask[self.num_sensors] = not bool(valid_sensors.any().item())

        return mask

    def _worker_sensor_mask_batch(self, obs, selected):
        """
        Batched version of _worker_sensor_mask().
        """

        pending = obs[:, :self.num_sensors] > 0.5
        valid_sensors = pending & ~selected

        mask = torch.zeros(
            (obs.shape[0], self.num_sensor_actions),
            dtype=torch.bool,
            device=self.device,
        )
        mask[:, :self.num_sensors] = valid_sensors
        mask[:, self.num_sensors] = ~valid_sensors.any(dim=1)

        return mask

    def _format_env_action(self, sensor_actions, power_actions):
        """
        Convert internal categorical actions to BaseUAVAoDTEnv action tuples.
        """

        env_action = []

        for m in range(self.num_uavs):
            sensor_action = int(sensor_actions[m].item())
            if self.continuous_power:
                power_unit = float(power_actions[m].item())
                power_action = self.power_min + power_unit * (self.power_max - self.power_min)
            else:
                power_action = int(power_actions[m].item())

            if sensor_action == self.num_sensors:
                sensor_id = -1
            else:
                sensor_id = sensor_action

            env_action.append((sensor_id, power_action))

        return env_action

    def save(self, path):
        """
        Save model parameters.
        """

        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Load model parameters.
        """

        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
