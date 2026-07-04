import numpy as np

from dt_uav_v2.envs.base_env import BaseUAVAoDTEnv


class WorkerEnv:
    """Slot-level wrapper used by the worker PPO and greedy worker."""

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

    def reset(self, seed=None, scenario=None):
        state = self.base_env.reset(seed=seed, scenario=scenario)
        obs = self._state_to_obs(state)
        self.obs_dim = len(obs)
        return obs

    def step(self, action):
        prev_avg_aodt = self.base_env.average_aodt()
        state, info = self.base_env.step_worker(action)
        reward = self._compute_reward(info, prev_avg_aodt)
        done = info["done"]
        obs = self._state_to_obs(state)
        info["reward"] = reward
        return obs, reward, done, info

    def _state_to_obs(self, state):
        q = state["Q"].astype(np.float32)
        u = (state["U"] / max(self.aoi_obs_norm, 1e-9)).astype(np.float32)
        w = (state["W"] / max(self.packet_size_max, 1e-9)).astype(np.float32)
        sensor_aoi = (state["sensor_aoi"] / max(self.aoi_obs_norm, 1e-9)).astype(np.float32)
        entity_aodt = (state["entity_aodt"] / max(self.aoi_obs_norm, 1e-9)).astype(np.float32)

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
        invalid_cost = self.invalid_action_penalty * float(info["invalid_count"]) / max(self.num_uavs, 1)
        wasted_cost = self.wasted_slot_penalty * float(info["wasted_count"]) / max(self.num_uavs, 1)

        avg_aodt = float(info["avg_aodt"])
        aodt_cost = avg_aodt / max(self.aodt_reward_scale, 1e-9)
        aodt_delta_bonus = (
            self.aodt_delta_weight
            * (float(prev_avg_aodt) - avg_aodt)
            / max(self.aodt_reward_scale, 1e-9)
        )
        return float(aodt_delta_bonus - aodt_cost - invalid_cost - wasted_cost)

    def _candidate_delay(self, sensor_id, uav_id, power_value):
        packet_size = float(self.base_env.W[sensor_id])
        entity_id = int(self.base_env.sensor_entity[sensor_id])
        dt_host = int(self.base_env.dt_hosts[entity_id])

        uplink_rate = self.base_env.uplink_rate(sensor_id, uav_id, power_value)
        uplink_delay = packet_size / max(uplink_rate, 1e-12)
        backhaul_delay = 0.0
        if uav_id != dt_host:
            backhaul_delay = packet_size / max(self.base_env.backhaul_rate(uav_id, dt_host), 1e-12)
        processing_delay = self.base_env.processing_delay(sensor_id)
        return float(uplink_delay + backhaul_delay + processing_delay)

    def sample_max_age_reduction_action(self):
        pending = np.where(self.base_env.Q > 0)[0]
        if len(pending) == 0:
            return [(-1, 0) for _ in range(self.num_uavs)]

        power_value = max(self.base_env.sensor_power_levels)
        pair_scores = []
        for sensor_id in pending:
            for uav_id in range(self.num_uavs):
                total_delay = self._candidate_delay(int(sensor_id), uav_id, power_value)
                if (
                    self.config.get("service_model", "abstract_same_step") == "require_within_slot"
                    and total_delay > self.base_env.slot_duration + 1e-6
                ):
                    continue
                new_aoi = self.base_env.U[sensor_id] + total_delay / max(self.base_env.slot_duration, 1e-9)
                gain = float(self.base_env.sensor_aoi[sensor_id] + 1.0 - new_aoi)
                pair_scores.append((gain, int(sensor_id), int(uav_id)))

        pair_scores.sort(key=lambda item: (-item[0], item[1], item[2]))
        action = [(-1, 0) for _ in range(self.num_uavs)]
        used_sensors = set()
        used_uavs = set()

        for gain, sensor_id, uav_id in pair_scores:
            if gain <= 0.0:
                continue
            if sensor_id in used_sensors or uav_id in used_uavs:
                continue
            action[uav_id] = (sensor_id, power_value if self.continuous_power else self.num_power_levels - 1)
            used_sensors.add(sensor_id)
            used_uavs.add(uav_id)

        return action

    def sample_proximity_greedy_action(self):
        action = [(-1, 0) for _ in range(self.num_uavs)]
        used_sensors = set()
        power_value = max(self.base_env.sensor_power_levels)

        for uav_id in range(self.num_uavs):
            best = None
            for sensor_id in np.where(self.base_env.Q > 0)[0]:
                sensor_id = int(sensor_id)
                if sensor_id in used_sensors:
                    continue
                total_delay = self._candidate_delay(sensor_id, uav_id, power_value)
                if (
                    self.config.get("service_model", "abstract_same_step") == "require_within_slot"
                    and total_delay > self.base_env.slot_duration + 1e-6
                ):
                    continue
                distance = float(
                    np.linalg.norm(
                        self.base_env.sensor_positions[sensor_id] - self.base_env.uav_positions[uav_id]
                    )
                )
                candidate = (distance, sensor_id)
                if best is None or candidate < best:
                    best = candidate
            if best is not None:
                _, sensor_id = best
                action[uav_id] = (
                    sensor_id,
                    power_value if self.continuous_power else self.num_power_levels - 1,
                )
                used_sensors.add(sensor_id)

        return action

    def sample_aodt_greedy_action(self):
        return self.sample_max_age_reduction_action()
