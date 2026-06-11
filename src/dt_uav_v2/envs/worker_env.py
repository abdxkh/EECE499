import numpy as np

from dt_uav_v2.envs.base_env import BaseUAVAoDTEnv


class WorkerEnv:
    """
    Thin Gym-style wrapper around BaseUAVAoDTEnv for worker training.

    This class does not train PPO and does not own any neural network logic.
    It only converts the simulator state into a flat numeric observation and
    computes the slot-level worker reward.
    """

    def __init__(self, config=None, base_env=None):
        self.base_env = base_env if base_env is not None else BaseUAVAoDTEnv(config=config)
        self.config = self.base_env.config

        self.invalid_action_penalty = self.config["invalid_action_penalty"]
        self.wasted_slot_penalty = self.config["wasted_slot_penalty"]
        self.aodt_reward_scale = self.config["aodt_reward_scale"]
        self.aodt_delta_weight = self.config["aodt_delta_weight"]

        self.packet_size_max = self.config["packet_size_max"]
        self.area_size = self.config["area_size"]
        self.episode_slots = self.config["episode_slots"]
        self.aoi_obs_norm = self.config["aoi_obs_norm"]

        self.num_uavs = self.base_env.M
        self.num_sensors = self.base_env.I
        self.num_power_levels = len(self.base_env.sensor_power_levels)
        self.continuous_power = self.config.get("worker_continuous_power", False)
        self.obs_dim = None

    def reset(self, seed=None):
        """
        Reset the underlying simulator and return the worker observation.
        """

        state = self.base_env.reset(seed=seed)
        obs = self._state_to_obs(state)
        self.obs_dim = len(obs)

        return obs

    def step(self, action):
        """
        Apply one worker action.

        Args:
            action:
                List of (sensor_id, power_index), one tuple per UAV.

        Returns:
            obs, reward, done, info
        """

        prev_avg_aodt = self.base_env.average_aodt()

        state, info = self.base_env.step_worker(action)
        reward = self._compute_reward(info, prev_avg_aodt)
        done = info["done"]
        obs = self._state_to_obs(state)

        info["reward"] = reward

        return obs, reward, done, info

    def _state_to_obs(self, state):
        """
        Convert the debug dictionary state into a flat numeric observation.

        Observation contents:
        - pending packet flags Q
        - packet waiting times U, normalized by AoI observation scale
        - packet sizes W, normalized by configured max packet size
        - sensor AoI, normalized by AoI observation scale
        - entity AoDT, normalized by AoI observation scale
        - sensor-to-UAV distances, normalized by area diagonal
        - DT host one-hot encoding, shape E x M
        - sensor-to-entity one-hot encoding, shape I x E
        - each sensor's DT-host one-hot encoding, shape I x M
        - current backhaul powers, normalized by max backhaul power, shape M
        """

        q = state["Q"].astype(np.float32)
        u = (state["U"] / max(self.aoi_obs_norm, 1e-9)).astype(np.float32)
        w = (state["W"] / max(self.packet_size_max, 1e-9)).astype(np.float32)

        sensor_aoi = (
            state["sensor_aoi"] / max(self.aoi_obs_norm, 1e-9)
        ).astype(np.float32)
        entity_aodt = (
            state["entity_aodt"] / max(self.aoi_obs_norm, 1e-9)
        ).astype(np.float32)

        distances = self.base_env.compute_sensor_uav_distances()
        area_diagonal = np.sqrt(2.0) * self.area_size
        distances = (distances / max(area_diagonal, 1e-9)).astype(np.float32).ravel()

        dt_hosts = state["dt_hosts"].astype(int)
        dt_host_one_hot = np.zeros((self.base_env.E, self.base_env.M), dtype=np.float32)
        dt_host_one_hot[np.arange(self.base_env.E), dt_hosts] = 1.0

        sensor_entity = state["sensor_entity"].astype(int)
        sensor_entity_one_hot = np.zeros((self.base_env.I, self.base_env.E), dtype=np.float32)
        sensor_entity_one_hot[np.arange(self.base_env.I), sensor_entity] = 1.0

        sensor_dt_hosts = dt_hosts[sensor_entity]
        sensor_dt_host_one_hot = np.zeros((self.base_env.I, self.base_env.M), dtype=np.float32)
        sensor_dt_host_one_hot[np.arange(self.base_env.I), sensor_dt_hosts] = 1.0

        backhaul_powers = (
            state["backhaul_powers"] / max(self.config["backhaul_power_max"], 1e-9)
        ).astype(np.float32)

        obs = np.concatenate(
            [
                q,
                u,
                w,
                sensor_aoi,
                entity_aodt,
                distances,
                dt_host_one_hot.ravel(),
                sensor_entity_one_hot.ravel(),
                sensor_dt_host_one_hot.ravel(),
                backhaul_powers,
            ]
        )

        return obs.astype(np.float32)

    def _compute_reward(self, info, prev_avg_aodt):
        """
        Compute slot-level worker reward.
        """

        avg_aodt = float(info["avg_aodt"])
        aodt_cost = avg_aodt / max(self.aodt_reward_scale, 1e-9)
        aodt_delta_bonus = (
            self.aodt_delta_weight
            * (float(prev_avg_aodt) - avg_aodt)
            / max(self.aodt_reward_scale, 1e-9)
        )
        invalid_cost = (
            self.invalid_action_penalty
            * float(info["invalid_count"])
            / max(self.num_uavs, 1)
        )
        wasted_cost = (
            self.wasted_slot_penalty
            * float(info["wasted_count"])
            / max(self.num_uavs, 1)
        )

        reward = aodt_delta_bonus - aodt_cost - invalid_cost - wasted_cost

        return float(reward)

    def sample_aodt_greedy_action(self):
        """
        Debug baseline: schedule the highest-AoI pending sensors.

        This is not used by PPO. It is useful for checking whether the learned
        worker is approaching a simple freshness-first policy.
        """

        pending = np.where(self.base_env.Q > 0)[0]
        action = []

        if len(pending) == 0:
            return [(-1, 0) for _ in range(self.num_uavs)]

        priorities = self.base_env.sensor_aoi[pending]
        ordered = pending[np.argsort(-priorities)]
        if self.continuous_power:
            max_power = max(self.base_env.sensor_power_levels)
        else:
            max_power = self.num_power_levels - 1

        for m in range(self.num_uavs):
            if m < len(ordered):
                action.append((int(ordered[m]), max_power))
            else:
                action.append((-1, 0))

        return action


if __name__ == "__main__":
    env = WorkerEnv()
    obs = env.reset()

    print("Worker environment reset successfully.")
    print("Observation shape:", obs.shape)
    print()

    for step in range(5):
        action = env.sample_aodt_greedy_action()
        obs, reward, done, info = env.step(action)

        print("Step:", step + 1)
        print("Observation shape:", obs.shape)
        print("Reward:", reward)
        print("Average AoDT:", info["avg_aodt"])
        print("Invalid actions:", info["invalid_count"])
        print("Wasted slots:", info["wasted_count"])
        print("Backhaul energy:", info["backhaul_energy"])
        print("Done:", done)
        print("-" * 50)

        if done:
            break
