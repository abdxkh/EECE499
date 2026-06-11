from pathlib import Path

import numpy as np

from dt_uav_v2.agents.ppo import PPOAgent
from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.base_env import BaseUAVAoDTEnv
from dt_uav_v2.envs.worker_env import WorkerEnv


class ManagerEnv:
    """
    Slow-timescale manager environment.

    The manager acts once every H worker slots. Its action sets UAV grid
    positions and DT placement. A frozen worker then handles the per-slot
    scheduling decisions for the whole manager window.
    """

    def __init__(
        self,
        config=None,
        worker_model_path="outputs/models/worker_continuous_final.pt",
        worker_policy="ppo",
    ):
        self.config = dict(CONFIG if config is None else config)

        self.base_env = BaseUAVAoDTEnv(config=self.config)
        self.worker_env = WorkerEnv(config=self.config, base_env=self.base_env)

        self.M = self.base_env.M
        self.E = self.base_env.E
        self.I = self.base_env.I

        self.H = self.config["manager_horizon"]
        self.episode_slots = self.config["episode_slots"]
        self.area_size = self.config["area_size"]
        self.aoi_obs_norm = self.config["aoi_obs_norm"]
        self.energy_budget = self.config["backhaul_energy_budget"]
        self.lyapunov_beta = self.config["lyapunov_beta"]
        self.optimize_backhaul_power = self.config.get("optimize_backhaul_power", False)
        self.backhaul_power_min = self.config.get("backhaul_power_min", self.config["backhaul_power"])
        self.backhaul_power_max = self.config.get("backhaul_power_max", self.config["backhaul_power"])

        self.grid_size = self.config.get("manager_grid_size", 4)
        self.grid_points = self._create_grid_points()
        self.num_grid_points = len(self.grid_points)

        self.worker_policy = worker_policy
        self.worker_model_path = Path(worker_model_path)
        self.worker_agent = None

        self.virtual_queues = np.zeros(self.M, dtype=np.float32)
        self.last_window_energy = np.zeros(self.M, dtype=np.float32)
        self.obs_dim = None

    def reset(self, seed=None):
        """
        Reset the manager episode and return manager observation.
        """

        self.base_env.reset(seed=seed)
        self.virtual_queues = np.zeros(self.M, dtype=np.float32)
        self.last_window_energy = np.zeros(self.M, dtype=np.float32)

        self._maybe_load_worker()

        obs = self._get_obs()
        self.obs_dim = len(obs)

        return obs

    def step(self, action):
        """
        Apply one manager action and run H worker slots.

        Action formats accepted:
        - dict with keys "uav_grid_indices" and "dt_hosts"
        - tuple/list: (uav_grid_indices, dt_hosts)
        """

        self._apply_manager_action(action)

        window_aodt = []
        window_tail_aodt = []
        window_energy = []
        total_invalid = 0
        total_wasted = 0
        worker_steps = 0
        done = False

        for _ in range(self.H):
            worker_action = self._select_worker_action()
            _, info = self.base_env.step_worker(worker_action)

            window_aodt.append(info["avg_aodt"])
            window_tail_aodt.append(self.base_env.tail_aodt())
            window_energy.append(info["backhaul_energy"])
            total_invalid += info["invalid_count"]
            total_wasted += info["wasted_count"]
            worker_steps += 1
            done = info["done"]

            if done:
                break

        avg_window_aodt = float(np.mean(window_aodt)) if window_aodt else 0.0
        tail_window_aodt = float(np.mean(window_tail_aodt)) if window_tail_aodt else 0.0
        avg_energy_per_uav = (
            np.mean(np.asarray(window_energy), axis=0)
            if window_energy
            else np.zeros(self.M)
        )

        self.last_window_energy = avg_energy_per_uav.astype(np.float32)

        energy_violation = avg_energy_per_uav - self.energy_budget
        positive_violation = np.maximum(energy_violation, 0.0)
        self.virtual_queues = np.maximum(
            0.0,
            self.virtual_queues + energy_violation,
        ).astype(np.float32)

        reward = self._compute_reward(avg_window_aodt, positive_violation)
        obs = self._get_obs()

        info = {
            "time": self.base_env.t,
            "worker_steps": worker_steps,
            "avg_window_aodt": avg_window_aodt,
            "tail_window_aodt": tail_window_aodt,
            "avg_energy_per_uav": avg_energy_per_uav.copy(),
            "energy_violation": energy_violation.copy(),
            "virtual_queues": self.virtual_queues.copy(),
            "invalid_count": total_invalid,
            "wasted_count": total_wasted,
            "dt_hosts": self.base_env.dt_hosts.copy(),
            "uav_positions": self.base_env.uav_positions.copy(),
            "backhaul_powers": self.base_env.backhaul_powers.copy(),
            "storage_used": self.base_env.compute_storage_used().copy(),
            "reward": reward,
            "done": done,
        }

        return obs, reward, done, info

    def sample_random_manager_action(self):
        """
        Random manager action for debugging.
        """

        uav_grid_indices = self.base_env.rng.integers(
            low=0,
            high=self.num_grid_points,
            size=self.M,
        )
        dt_hosts = self.base_env.rng.integers(low=0, high=self.M, size=self.E)

        action = {
            "uav_grid_indices": uav_grid_indices.astype(int),
            "dt_hosts": dt_hosts.astype(int),
        }

        if self.optimize_backhaul_power:
            action["backhaul_powers"] = self.base_env.rng.uniform(
                low=self.backhaul_power_min,
                high=self.backhaul_power_max,
                size=self.M,
            ).astype(np.float32)

        return action

    def _create_grid_points(self):
        coords = np.linspace(0.0, self.area_size, self.grid_size)
        points = []

        for x in coords:
            for y in coords:
                points.append([x, y])

        return np.asarray(points, dtype=np.float32)

    def _maybe_load_worker(self):
        if self.worker_policy != "ppo":
            return

        if self.worker_agent is not None:
            return

        worker_obs = self.worker_env._state_to_obs(self.base_env.get_basic_state())
        self.worker_env.obs_dim = len(worker_obs)

        self.worker_agent = PPOAgent(
            obs_dim=self.worker_env.obs_dim,
            num_uavs=self.worker_env.num_uavs,
            num_sensors=self.worker_env.num_sensors,
            num_power_levels=self.worker_env.num_power_levels,
            lr=self.config["worker_lr"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_eps=self.config["clip_eps"],
            mask_worker_actions=True,
            worker_freshness_bias=self.config["worker_freshness_bias"],
            force_max_power=self.config["worker_force_max_power"],
            continuous_power=self.config.get("worker_continuous_power", False),
            power_min=min(self.config["sensor_power_levels"]),
            power_max=max(self.config["sensor_power_levels"]),
        )

        if not self.worker_model_path.exists():
            raise FileNotFoundError(f"Worker model not found: {self.worker_model_path}")

        self.worker_agent.load(self.worker_model_path)

    def _select_worker_action(self):
        if self.worker_policy == "greedy":
            return self.worker_env.sample_aodt_greedy_action()

        if self.worker_policy != "ppo":
            raise ValueError(f"Unknown worker policy: {self.worker_policy}")

        worker_obs = self.worker_env._state_to_obs(self.base_env.get_basic_state())
        action, _, _, _ = self.worker_agent.select_action(
            worker_obs,
            deterministic=True,
        )

        return action

    def _apply_manager_action(self, action):
        if isinstance(action, dict):
            uav_grid_indices = action["uav_grid_indices"]
            dt_hosts = action["dt_hosts"]
            backhaul_powers = action.get("backhaul_powers", None)
        else:
            if len(action) == 3:
                uav_grid_indices, dt_hosts, backhaul_powers = action
            else:
                uav_grid_indices, dt_hosts = action
                backhaul_powers = None

        uav_grid_indices = np.asarray(uav_grid_indices, dtype=int)
        dt_hosts = np.asarray(dt_hosts, dtype=int)

        if len(uav_grid_indices) != self.M:
            raise ValueError("Manager action must provide one grid index per UAV.")

        if len(dt_hosts) != self.E:
            raise ValueError("Manager action must provide one DT host per entity.")

        uav_grid_indices = np.clip(uav_grid_indices, 0, self.num_grid_points - 1)
        dt_hosts = np.clip(dt_hosts, 0, self.M - 1)

        self.base_env.uav_positions = self.grid_points[uav_grid_indices].copy()
        self.base_env.dt_hosts = self.base_env._repair_dt_storage(dt_hosts.copy())

        if self.optimize_backhaul_power:
            if backhaul_powers is None:
                backhaul_powers = np.ones(self.M) * self.config["backhaul_power"]

            backhaul_powers = np.asarray(backhaul_powers, dtype=float)
            if len(backhaul_powers) != self.M:
                raise ValueError("Manager action must provide one backhaul power per UAV.")

            self.base_env.backhaul_powers = np.clip(
                backhaul_powers,
                self.backhaul_power_min,
                self.backhaul_power_max,
            ).astype(float)

    def _compute_reward(self, avg_window_aodt, positive_violation):
        aodt_cost = avg_window_aodt / max(self.aoi_obs_norm, 1e-9)
        queue_cost = self.lyapunov_beta * float(
            np.sum(self.virtual_queues * positive_violation)
        ) / max(self.energy_budget, 1e-9)

        return -float(aodt_cost + queue_cost)

    def _get_obs(self):
        state = self.base_env.get_basic_state()

        time_fraction = np.asarray(
            [state["time"] / max(self.episode_slots, 1)],
            dtype=np.float32,
        )
        uav_positions = (state["uav_positions"] / max(self.area_size, 1e-9)).astype(
            np.float32
        ).ravel()

        dt_hosts = state["dt_hosts"].astype(int)
        dt_host_one_hot = np.zeros((self.E, self.M), dtype=np.float32)
        dt_host_one_hot[np.arange(self.E), dt_hosts] = 1.0

        storage_used = state["storage_used"] / np.maximum(
            self.base_env.uav_storage_capacity,
            1e-9,
        )
        storage_used = storage_used.astype(np.float32)

        entity_aodt = (
            state["entity_aodt"] / max(self.aoi_obs_norm, 1e-9)
        ).astype(np.float32)
        virtual_queues = (
            self.virtual_queues / max(self.energy_budget, 1e-9)
        ).astype(np.float32)
        last_energy = (
            self.last_window_energy / max(self.energy_budget, 1e-9)
        ).astype(np.float32)
        backhaul_powers = (
            state["backhaul_powers"] / max(self.backhaul_power_max, 1e-9)
        ).astype(np.float32)

        obs_parts = [
            time_fraction,
            uav_positions,
            dt_host_one_hot.ravel(),
            storage_used,
            entity_aodt,
            virtual_queues,
            last_energy,
        ]

        if self.optimize_backhaul_power:
            obs_parts.append(backhaul_powers)

        obs = np.concatenate(obs_parts)

        return obs.astype(np.float32)


if __name__ == "__main__":
    env = ManagerEnv(worker_policy="ppo")
    obs = env.reset(seed=CONFIG["seed"])

    print("Manager environment reset successfully.")
    print("Observation shape:", obs.shape)
    print("Grid points:", env.num_grid_points)
    print("Manager horizon:", env.H)
    print()

    for step in range(3):
        action = env.sample_random_manager_action()
        obs, reward, done, info = env.step(action)

        print("Manager step:", step + 1)
        print("Reward:", reward)
        print("Average window AoDT:", info["avg_window_aodt"])
        print("Tail window AoDT:", info["tail_window_aodt"])
        print("Average energy per UAV:", info["avg_energy_per_uav"])
        print("Virtual queues:", info["virtual_queues"])
        print("Invalid worker actions:", info["invalid_count"])
        print("Wasted worker actions:", info["wasted_count"])
        print("Done:", done)
        print("-" * 60)

        if done:
            break
