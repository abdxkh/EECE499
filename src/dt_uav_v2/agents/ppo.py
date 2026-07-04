from dataclasses import dataclass, field
import subprocess

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
        power_mode="learned_beta",
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.num_uavs = num_uavs
        self.num_sensors = num_sensors
        self.num_sensor_actions = num_sensors + 1
        self.num_power_levels = num_power_levels
        self.continuous_power = continuous_power
        self.power_mode = power_mode

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.sensor_head = nn.Linear(hidden_dim, num_uavs * self.num_sensor_actions)
        if self.continuous_power and self.power_mode == "learned_beta":
            self.power_head = nn.Linear(hidden_dim, num_uavs * 2)
        elif not self.continuous_power:
            self.power_head = nn.Linear(hidden_dim, num_uavs * num_power_levels)
        else:
            self.power_head = None
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

        power_output = None
        if self.power_head is not None:
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
        num_entities=None,
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
        power_mode=None,
        continuous_power=False,
        power_min=None,
        power_max=None,
        slot_duration=1.0,
        bandwidth_access=1.0,
        bandwidth_backhaul=1.0,
        noise_power=1.0,
        pathloss_ref=1e-3,
        pathloss_exp=2.0,
        cpu_cycles_per_bit=1000.0,
        cpu_rate=1e9,
        packet_size_max=1.0,
        area_size=1.0,
        backhaul_power_max=1.0,
        backhaul_power_min=0.0,
        delay_tolerance=1e-6,
        service_model="abstract_same_step",
        device=None,
    ):
        self.obs_dim = obs_dim
        self.num_uavs = num_uavs
        self.num_sensors = num_sensors
        self.num_sensor_actions = num_sensors + 1
        self.num_power_levels = num_power_levels
        self.num_entities = int(num_entities) if num_entities is not None else None

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
        if power_mode is None:
            if force_max_power:
                power_mode = "fixed_max"
            elif continuous_power:
                power_mode = "learned_beta"
            else:
                power_mode = "discrete"
        self.power_mode = power_mode
        self.force_max_power = power_mode == "fixed_max"
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
        self.slot_duration = float(slot_duration)
        self.bandwidth_access = float(bandwidth_access)
        self.bandwidth_backhaul = float(bandwidth_backhaul)
        self.noise_power = float(noise_power)
        self.pathloss_ref = float(pathloss_ref)
        self.pathloss_exp = float(pathloss_exp)
        self.cpu_cycles_per_bit = float(cpu_cycles_per_bit)
        self.cpu_rate = float(cpu_rate)
        self.packet_size_max = float(packet_size_max)
        self.area_size = float(area_size)
        self.backhaul_power_max = float(backhaul_power_max)
        self.backhaul_power_min = float(backhaul_power_min)
        self.delay_tolerance = float(delay_tolerance)
        self.service_model = service_model
        self.area_diagonal = np.sqrt(2.0) * self.area_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(
            obs_dim=obs_dim,
            num_uavs=num_uavs,
            num_sensors=num_sensors,
            num_power_levels=num_power_levels,
            hidden_dim=hidden_dim,
            continuous_power=continuous_power,
            power_mode=self.power_mode,
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

            if self.continuous_power:
                if self.power_mode == "fixed_max":
                    power_actions = torch.ones(
                        self.num_uavs,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    power_log_prob = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)
                elif self.power_mode == "fixed_mid":
                    power_actions = torch.full(
                        (self.num_uavs,),
                        0.5,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    power_log_prob = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)
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
                    power_log_prob = power_dist.log_prob(power_actions)
            elif self.force_max_power:
                power_actions = torch.full(
                    (self.num_uavs,),
                    self.num_power_levels - 1,
                    dtype=torch.float32,
                    device=self.device,
                )
                power_log_prob = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)
            else:
                power_dist = Categorical(logits=power_output[0])
                if deterministic:
                    power_actions = torch.argmax(power_output[0], dim=-1)
                else:
                    power_actions = power_dist.sample()
                power_log_prob = power_dist.log_prob(power_actions)

            sensor_actions, sensor_log_prob, sensor_entropy = self._sample_sensor_actions(
                sensor_logits[0],
                obs_tensor[0],
                power_actions,
                deterministic=deterministic,
            )

            active_power_mask = sensor_actions != self.num_sensors
            if self.continuous_power and self.power_mode == "learned_beta":
                log_prob = sensor_log_prob + power_log_prob[active_power_mask].sum()
            elif (not self.continuous_power) and (not self.force_max_power):
                log_prob = sensor_log_prob + power_log_prob.sum()
            else:
                log_prob = sensor_log_prob

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

        with torch.no_grad():
            pre_update_log_probs, _, pre_update_values = self._evaluate_actions(obs, actions)
            approx_kl = float(torch.mean(old_log_probs - pre_update_log_probs).item())
            pre_update_ratio = torch.exp(pre_update_log_probs - old_log_probs)
            clip_fraction = float(
                torch.mean(
                    (
                        (pre_update_ratio < (1.0 - self.clip_eps))
                        | (pre_update_ratio > (1.0 + self.clip_eps))
                    ).float()
                ).item()
            )
            returns_var = torch.var(returns, unbiased=False)
            if float(returns_var.item()) > 1e-8:
                explained_variance = float(
                    1.0
                    - (
                        torch.var(returns - pre_update_values, unbiased=False)
                        / returns_var
                    ).item()
                )
            else:
                explained_variance = 0.0

        num_steps = obs.shape[0]
        final_stats = {}
        grad_norm_total = 0.0
        batch_count = 0

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
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                grad_norm_total += float(grad_norm)
                batch_count += 1

                final_stats = {
                    "loss": float(loss.item()),
                    "actor_loss": float(actor_loss.item()),
                    "critic_loss": float(critic_loss.item()),
                    "entropy": float(entropy_loss.item()),
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "explained_variance": explained_variance,
                    "grad_norm": float(grad_norm_total / max(batch_count, 1)),
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
                power_actions,
            )
        else:
            sensor_dist = Categorical(logits=sensor_logits)
            sensor_log_probs = sensor_dist.log_prob(sensor_actions).sum(dim=1)
            sensor_entropy = sensor_dist.entropy().sum(dim=1)

        if self.continuous_power:
            active_power_mask = sensor_actions != self.num_sensors
            if self.power_mode in {"fixed_max", "fixed_mid"}:
                log_probs = sensor_log_probs
                entropy = sensor_entropy
            else:
                power_dist = Beta(power_output[:, :, 0], power_output[:, :, 1])
                power_actions = torch.clamp(power_actions, 1e-6, 1.0 - 1e-6)
                power_log_probs = power_dist.log_prob(power_actions) * active_power_mask.float()
                power_entropy = power_dist.entropy() * active_power_mask.float()
                log_probs = sensor_log_probs + power_log_probs.sum(dim=1)
                entropy = sensor_entropy + power_entropy.sum(dim=1)
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

    def _sample_sensor_actions(self, sensor_logits, obs, power_actions, deterministic=False):
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
            mask = self._worker_sensor_mask(obs, selected, m, power_actions[m])
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

    def _evaluate_masked_sensor_actions(self, sensor_logits, obs, sensor_actions, power_actions):
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
            mask = self._worker_sensor_mask_batch(obs, selected, m, power_actions[:, m])
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

    def _worker_sensor_mask(self, obs, selected, uav_index, power_action):
        """
        Build a one-step worker sensor mask for one observation.

        The observation starts with Q, so obs[:num_sensors] gives pending flags.
        Idle is allowed only when no valid unselected pending sensor remains.
        """

        pending = obs[:self.num_sensors] > 0.5
        valid_sensors = pending & ~selected
        if self.service_model == "require_within_slot":
            feasible_delay = self._worker_delay_feasible_mask(obs, uav_index, power_action)
            valid_sensors = valid_sensors & feasible_delay

        mask = torch.zeros(self.num_sensor_actions, dtype=torch.bool, device=self.device)
        mask[:self.num_sensors] = valid_sensors
        mask[self.num_sensors] = True

        return mask

    def _worker_sensor_mask_batch(self, obs, selected, uav_index, power_actions):
        """
        Batched version of _worker_sensor_mask().
        """

        pending = obs[:, :self.num_sensors] > 0.5
        valid_sensors = pending & ~selected
        if self.service_model == "require_within_slot":
            feasible_delay = self._worker_delay_feasible_mask_batch(obs, uav_index, power_actions)
            valid_sensors = valid_sensors & feasible_delay

        mask = torch.zeros(
            (obs.shape[0], self.num_sensor_actions),
            dtype=torch.bool,
            device=self.device,
        )
        mask[:, :self.num_sensors] = valid_sensors
        mask[:, self.num_sensors] = True

        return mask

    def _effective_uplink_power(self, power_action):
        if self.continuous_power:
            if self.power_mode == "fixed_max":
                return torch.as_tensor(self.power_max, dtype=torch.float32, device=self.device)
            if self.power_mode == "fixed_mid":
                return torch.as_tensor(
                    0.5 * (self.power_min + self.power_max),
                    dtype=torch.float32,
                    device=self.device,
                )
            return self.power_min + power_action * (self.power_max - self.power_min)
        if self.force_max_power:
            return torch.as_tensor(self.power_max, dtype=torch.float32, device=self.device)
        raise RuntimeError("Delay-feasibility masking is only implemented for fixed worker power modes and learned_beta.")

    def _decode_worker_obs(self, obs):
        if self.num_entities is None:
            raise RuntimeError("Worker delay-feasibility masking requires num_entities.")
        idx = 0
        q = obs[idx:idx + self.num_sensors]
        idx += self.num_sensors
        _u = obs[idx:idx + self.num_sensors]
        idx += self.num_sensors
        w = obs[idx:idx + self.num_sensors] * self.packet_size_max
        idx += self.num_sensors
        idx += self.num_sensors  # sensor_aoi
        idx += self.num_entities  # entity_aodt
        distances = obs[idx:idx + self.num_sensors * self.num_uavs].view(self.num_sensors, self.num_uavs)
        distances = distances * self.area_diagonal
        idx += self.num_sensors * self.num_uavs
        idx += self.num_entities * self.num_uavs  # entity host one-hot
        idx += self.num_sensors * self.num_entities  # sensor-entity one-hot
        sensor_dt_host_one_hot = obs[idx:idx + self.num_sensors * self.num_uavs].view(
            self.num_sensors,
            self.num_uavs,
        )
        idx += self.num_sensors * self.num_uavs
        backhaul_powers = obs[idx:idx + self.num_uavs] * self.backhaul_power_max
        dt_host_indices = torch.argmax(sensor_dt_host_one_hot, dim=1)
        return q, w, distances, dt_host_indices, backhaul_powers

    def _channel_gain(self, distance):
        return self.pathloss_ref / torch.pow(distance + 1.0, self.pathloss_exp)

    def _backhaul_bandwidth_per_link(self):
        return self.bandwidth_backhaul / max(self.num_uavs * (self.num_uavs - 1), 1)

    def _worker_delay_feasible_mask(self, obs, uav_index, power_action):
        q, w, distances, dt_host_indices, backhaul_powers = self._decode_worker_obs(obs)
        power = self._effective_uplink_power(power_action)
        sensor_distances = distances[:, uav_index]
        uplink_rates = self.bandwidth_access * torch.log2(
            1.0 + (power * self._channel_gain(sensor_distances)) / max(self.noise_power, 1e-12)
        )
        uplink_rates = torch.clamp(uplink_rates, min=1e-9)
        uplink_delay = w / uplink_rates

        host_distances = torch.zeros_like(sensor_distances)
        backhaul_delay = torch.zeros_like(sensor_distances)
        cross = dt_host_indices != uav_index
        if cross.any():
            dt_hosts = dt_host_indices[cross]
            host_distances[cross] = distances[cross, dt_hosts]
            bh_power = backhaul_powers[uav_index]
            bh_rates = self._backhaul_bandwidth_per_link() * torch.log2(
                1.0 + (bh_power * self._channel_gain(host_distances[cross])) / max(self.noise_power, 1e-12)
            )
            bh_rates = torch.clamp(bh_rates, min=1e-9)
            backhaul_delay[cross] = w[cross] / bh_rates

        processing_delay = (w * self.cpu_cycles_per_bit) / max(self.cpu_rate, 1e-12)
        total_delay = uplink_delay + backhaul_delay + processing_delay
        pending = q > 0.5
        return pending & (total_delay <= (self.slot_duration + self.delay_tolerance))

    def _worker_delay_feasible_mask_batch(self, obs, uav_index, power_actions):
        masks = []
        for row_idx in range(obs.shape[0]):
            masks.append(
                self._worker_delay_feasible_mask(
                    obs[row_idx],
                    uav_index,
                    power_actions[row_idx],
                )
            )
        return torch.stack(masks, dim=0)

    def _format_env_action(self, sensor_actions, power_actions):
        """
        Convert internal categorical actions to BaseUAVAoDTEnv action tuples.
        """

        env_action = []

        for m in range(self.num_uavs):
            sensor_action = int(sensor_actions[m].item())
            if self.continuous_power:
                if self.power_mode == "fixed_max":
                    power_action = self.power_max
                elif self.power_mode == "fixed_mid":
                    power_action = 0.5 * (self.power_min + self.power_max)
                else:
                    power_unit = float(power_actions[m].item())
                    power_action = self.power_min + power_unit * (self.power_max - self.power_min)
            else:
                power_action = int(power_actions[m].item())

            if sensor_action == self.num_sensors:
                sensor_id = -1
                power_action = 0.0
            else:
                sensor_id = sensor_action

            env_action.append((sensor_id, power_action))

        return env_action

    def worker_feasible_real_mask(self, obs, deterministic=True):
        """
        Return the per-UAV feasible real-sensor mask under the current policy's
        power mode and service model, before duplicate suppression.
        """

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            _, power_output, _ = self.model(obs_tensor)
            if self.continuous_power:
                if self.power_mode == "fixed_max":
                    power_actions = torch.ones(self.num_uavs, dtype=torch.float32, device=self.device)
                elif self.power_mode == "fixed_mid":
                    power_actions = torch.full((self.num_uavs,), 0.5, dtype=torch.float32, device=self.device)
                else:
                    if deterministic:
                        power_actions = power_output[0, :, 0] / (power_output[0, :, 0] + power_output[0, :, 1])
                    else:
                        power_actions = Beta(power_output[0, :, 0], power_output[0, :, 1]).sample()
            elif self.force_max_power:
                power_actions = torch.full(
                    (self.num_uavs,),
                    self.num_power_levels - 1,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                power_actions = torch.argmax(power_output[0], dim=-1) if deterministic else Categorical(logits=power_output[0]).sample()

            selected = torch.zeros(self.num_sensors, dtype=torch.bool, device=self.device)
            masks = []
            for m in range(self.num_uavs):
                mask = self._worker_sensor_mask(obs_tensor[0], selected, m, power_actions[m])
                masks.append(mask[:self.num_sensors])
            return torch.stack(masks, dim=0).cpu().numpy().astype(bool)

    def save(self, path):
        """
        Save model parameters.
        """

        self.save_checkpoint(path)

    def load(self, path):
        """
        Load model parameters.
        """

        self.load_checkpoint(path)

    def checkpoint_metadata(self, extra_metadata=None):
        metadata = {
            "obs_dim": int(self.obs_dim),
            "num_uavs": int(self.num_uavs),
            "num_sensors": int(self.num_sensors),
            "num_entities": int(self.num_entities) if self.num_entities is not None else None,
            "num_power_levels": int(self.num_power_levels),
            "power_mode": self.power_mode,
            "continuous_power": bool(self.continuous_power),
            "hidden_dim": int(self.model.backbone[0].out_features),
            "service_model": self.service_model,
            "architecture_variant": f"worker_{self.power_mode}",
            "device": str(self.device),
            "git_commit": self._git_commit_hash(),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    def save_checkpoint(self, path, extra_metadata=None):
        payload = {
            "format_version": 2,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": self.checkpoint_metadata(extra_metadata),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location=self.device)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            metadata = payload.get("metadata", {})
            saved_obs_dim = metadata.get("obs_dim", self.obs_dim)
            if int(saved_obs_dim) != int(self.obs_dim):
                raise RuntimeError(
                    f"Incompatible checkpoint observation dimension: saved {saved_obs_dim}, current {self.obs_dim}."
                )
            saved_power_mode = metadata.get("power_mode", self.power_mode)
            if saved_power_mode != self.power_mode:
                raise RuntimeError(
                    f"Incompatible checkpoint power mode: saved {saved_power_mode}, current {self.power_mode}."
                )
            saved_service_model = metadata.get("service_model", self.service_model)
            if saved_service_model != self.service_model:
                raise RuntimeError(
                    f"Incompatible checkpoint service model: saved {saved_service_model}, current {self.service_model}."
                )
            self.model.load_state_dict(payload["model_state_dict"])
            optimizer_state = payload.get("optimizer_state_dict", None)
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
        else:
            self.model.load_state_dict(payload)
        self.model.to(self.device)

    def _git_commit_hash(self):
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            )
        except Exception:
            return None
