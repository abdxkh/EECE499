from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta, Categorical
import torch.nn.functional as F


class ManagerActorCritic(nn.Module):
    """
    Actor-critic network for slow manager actions.

    The manager action has two multi-categorical parts:
    - one UAV grid index per UAV
    - one DT host UAV per entity
    """

    def __init__(
        self,
        obs_dim,
        num_uavs,
        num_entities,
        num_grid_points,
        hidden_dim=128,
        optimize_backhaul_power=False,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.num_uavs = num_uavs
        self.num_entities = num_entities
        self.num_grid_points = num_grid_points
        self.optimize_backhaul_power = optimize_backhaul_power

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.grid_head = nn.Linear(hidden_dim, num_uavs * num_grid_points)
        self.host_head = nn.Linear(hidden_dim, num_entities * num_uavs)
        if self.optimize_backhaul_power:
            self.backhaul_power_head = nn.Linear(hidden_dim, num_uavs * 2)
        else:
            self.backhaul_power_head = None
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.backbone(obs)

        grid_logits = self.grid_head(features)
        grid_logits = grid_logits.view(-1, self.num_uavs, self.num_grid_points)

        host_logits = self.host_head(features)
        host_logits = host_logits.view(-1, self.num_entities, self.num_uavs)

        backhaul_power_output = None
        if self.optimize_backhaul_power:
            backhaul_power_output = self.backhaul_power_head(features)
            backhaul_power_output = backhaul_power_output.view(-1, self.num_uavs, 2)
            backhaul_power_output = F.softplus(backhaul_power_output) + 1.0

        value = self.value_head(features).squeeze(-1)

        return grid_logits, host_logits, backhaul_power_output, value


@dataclass
class ManagerRolloutMemory:
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


class ManagerPPOAgent:
    """
    PPO agent for manager deployment and DT placement decisions.
    """

    def __init__(
        self,
        obs_dim,
        num_uavs,
        num_entities,
        num_grid_points,
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
        optimize_backhaul_power=False,
        backhaul_power_min=0.1,
        backhaul_power_max=1.0,
        device=None,
    ):
        self.obs_dim = obs_dim
        self.num_uavs = num_uavs
        self.num_entities = num_entities
        self.num_grid_points = num_grid_points
        self.optimize_backhaul_power = optimize_backhaul_power
        self.backhaul_power_min = float(backhaul_power_min)
        self.backhaul_power_max = float(backhaul_power_max)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ManagerActorCritic(
            obs_dim=obs_dim,
            num_uavs=num_uavs,
            num_entities=num_entities,
            num_grid_points=num_grid_points,
            hidden_dim=hidden_dim,
            optimize_backhaul_power=optimize_backhaul_power,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, obs, deterministic=False):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)

        with torch.no_grad():
            grid_logits, host_logits, backhaul_power_output, value = self.model(obs_tensor)
            grid_dist = Categorical(logits=grid_logits[0])
            host_dist = Categorical(logits=host_logits[0])

            if deterministic:
                grid_actions = torch.argmax(grid_logits[0], dim=-1)
                host_actions = torch.argmax(host_logits[0], dim=-1)
            else:
                grid_actions = grid_dist.sample()
                host_actions = host_dist.sample()

            log_prob = grid_dist.log_prob(grid_actions).sum()
            log_prob = log_prob + host_dist.log_prob(host_actions).sum()

            backhaul_power_actions = None
            if self.optimize_backhaul_power:
                backhaul_power_dist = Beta(
                    backhaul_power_output[0, :, 0],
                    backhaul_power_output[0, :, 1],
                )
                if deterministic:
                    backhaul_power_actions = (
                        backhaul_power_output[0, :, 0]
                        / (
                            backhaul_power_output[0, :, 0]
                            + backhaul_power_output[0, :, 1]
                        )
                    )
                else:
                    backhaul_power_actions = backhaul_power_dist.sample()
                log_prob = log_prob + backhaul_power_dist.log_prob(backhaul_power_actions).sum()

        env_action = {
            "uav_grid_indices": grid_actions.cpu().numpy().astype(int),
            "dt_hosts": host_actions.cpu().numpy().astype(int),
        }
        action_parts = [grid_actions.float(), host_actions.float()]

        if self.optimize_backhaul_power:
            power_unit = backhaul_power_actions.cpu().numpy()
            env_action["backhaul_powers"] = (
                self.backhaul_power_min
                + power_unit * (self.backhaul_power_max - self.backhaul_power_min)
            ).astype(float)
            action_parts.append(backhaul_power_actions.float())

        action_indices = torch.cat(action_parts, dim=0)

        return (
            env_action,
            action_indices.cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def compute_returns_and_advantages(self, rewards, dones, values, last_value=0.0):
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae

        returns = advantages + values

        return returns.astype(np.float32), advantages.astype(np.float32)

    def update(self, memory, last_value=0.0):
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
        grid_logits, host_logits, backhaul_power_output, values = self.model(obs)

        grid_actions = actions[:, :self.num_uavs].long()
        host_actions = actions[:, self.num_uavs:self.num_uavs + self.num_entities].long()

        grid_dist = Categorical(logits=grid_logits)
        host_dist = Categorical(logits=host_logits)

        log_probs = grid_dist.log_prob(grid_actions).sum(dim=1)
        log_probs = log_probs + host_dist.log_prob(host_actions).sum(dim=1)

        entropy = grid_dist.entropy().sum(dim=1)
        entropy = entropy + host_dist.entropy().sum(dim=1)

        if self.optimize_backhaul_power:
            start = self.num_uavs + self.num_entities
            backhaul_power_actions = actions[:, start:start + self.num_uavs]
            backhaul_power_actions = torch.clamp(backhaul_power_actions, 1e-6, 1.0 - 1e-6)
            backhaul_power_dist = Beta(
                backhaul_power_output[:, :, 0],
                backhaul_power_output[:, :, 1],
            )
            log_probs = log_probs + backhaul_power_dist.log_prob(backhaul_power_actions).sum(dim=1)
            entropy = entropy + backhaul_power_dist.entropy().sum(dim=1)

        return log_probs, entropy, values

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
