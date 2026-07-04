from pathlib import Path
import time

import numpy as np
import torch

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
        self.manager_reward_mode = self.config.get("manager_reward_mode", "queue_weighted_energy")
        self.manager_aodt_weight = float(self.config.get("manager_aodt_weight", 1.0))
        self.manager_energy_weight = float(self.config.get("manager_energy_weight", 1.0))
        self.host_action_mode = self.config.get("manager_host_action_mode", "feasible_enum")
        self.optimize_backhaul_power = self.config.get("optimize_backhaul_power", False)
        self.backhaul_power_min = self.config.get("backhaul_power_min", self.config["backhaul_power"])
        self.backhaul_power_max = self.config.get("backhaul_power_max", self.config["backhaul_power"])

        self.grid_size = self.config.get("manager_grid_size", 4)
        self.grid_points = self._create_grid_points()
        self.num_grid_points = len(self.grid_points)
        self.total_dt_assignments = len(self.base_env.enumerate_all_dt_assignments())

        self.worker_policy = worker_policy
        self.worker_model_path = Path(worker_model_path)
        self.worker_agent = None

        self.virtual_queues = np.zeros(self.M, dtype=np.float32)
        self.last_window_energy = np.zeros(self.M, dtype=np.float32)
        self.last_action_diagnostics = {}
        self.obs_dim = None
        self.transition_count = 0

    def reset(self, seed=None, scenario=None):
        """
        Reset the manager episode and return manager observation.
        """

        self.base_env.reset(seed=seed, scenario=scenario)
        self.virtual_queues = np.zeros(self.M, dtype=np.float32)
        self.last_window_energy = np.zeros(self.M, dtype=np.float32)
        self.last_action_diagnostics = {
            "uav_switches": 0,
            "uav_switch_fraction": 0.0,
            "changed_uavs_this_transition": 0.0,
            "movement_distance": 0.0,
            "movement_distance_per_uav_transition": 0.0,
            "dt_switches": 0,
            "dt_host_switch_fraction": 0.0,
            "rehosted_entities_this_transition": 0.0,
            "dt_switches_per_entity": np.zeros(self.E, dtype=np.float32),
            "executed_dt_assignment_index": self.current_dt_assignment_index(),
        }
        self.transition_count = 0

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

        action_diag = self._apply_manager_action(action)

        window_aodt = []
        window_tail_aodt = []
        window_energy = []
        worker_slot_delay_stats = []
        total_invalid = 0
        total_wasted = 0
        worker_steps = 0
        worker_idle_count = 0
        worker_action_count = 0
        worker_select_time_total = 0.0
        done = False
        transmission_records = []

        for _ in range(self.H):
            t_worker = time.perf_counter()
            worker_action = self._select_worker_action()
            worker_select_time_total += time.perf_counter() - t_worker
            worker_idle_count += sum(1 for sensor_id, _ in worker_action if int(sensor_id) == -1)
            worker_action_count += len(worker_action)
            _, info = self.base_env.step_worker(worker_action)

            window_aodt.append(info["avg_aodt"])
            window_tail_aodt.append(self.base_env.tail_aodt())
            window_energy.append(info["backhaul_energy"])
            served_indices = np.where(info["served"] > 0.5)[0]
            if len(served_indices) > 0:
                worker_slot_delay_stats.extend(info["total_delay"][served_indices].tolist())
            transmission_records.extend(info.get("transmission_records", []))
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
            else np.zeros(self.M, dtype=np.float32)
        )

        self.last_window_energy = avg_energy_per_uav.astype(np.float32)

        old_queues = self.virtual_queues.copy()
        energy_violation = avg_energy_per_uav - self.energy_budget
        positive_violation = np.maximum(energy_violation, 0.0)
        reward, reward_terms = self._compute_reward(
            avg_window_aodt=avg_window_aodt,
            avg_energy_per_uav=avg_energy_per_uav,
            old_queues=old_queues,
        )
        self.virtual_queues = np.maximum(
            0.0,
            self.virtual_queues + energy_violation,
        ).astype(np.float32)
        obs = self._get_obs()

        info = {
            "time": self.base_env.t,
            "worker_steps": worker_steps,
            "worker_idle_count": int(worker_idle_count),
            "worker_action_count": int(worker_action_count),
            "worker_select_time_total": float(worker_select_time_total),
            "worker_select_time_mean": float(worker_select_time_total / max(worker_steps, 1)),
            "avg_window_aodt": avg_window_aodt,
            "tail_window_aodt": tail_window_aodt,
            "avg_energy_per_uav": avg_energy_per_uav.copy(),
            "max_energy_per_uav": float(np.max(avg_energy_per_uav)),
            "mean_energy_per_uav": float(np.mean(avg_energy_per_uav)),
            "energy_violation": energy_violation.copy(),
            "signed_violation_mean": float(np.mean(energy_violation)),
            "positive_violation": positive_violation.copy(),
            "positive_violation_mean": float(np.mean(positive_violation)),
            "old_virtual_queues": old_queues.copy(),
            "virtual_queues": self.virtual_queues.copy(),
            "invalid_count": total_invalid,
            "wasted_count": total_wasted,
            "dt_hosts": self.base_env.dt_hosts.copy(),
            "uav_positions": self.base_env.uav_positions.copy(),
            "backhaul_powers": self.base_env.backhaul_powers.copy(),
            "storage_used": self.base_env.compute_storage_used().copy(),
            "worker_slot_delay_stats": list(worker_slot_delay_stats),
            "transmission_records": transmission_records,
            "reward_terms": reward_terms,
            "uav_switches": int(action_diag["uav_switches"]),
            "uav_switch_fraction": float(action_diag["uav_switch_fraction"]),
            "changed_uavs_this_transition": float(action_diag["changed_uavs_this_transition"]),
            "movement_distance": float(action_diag["movement_distance"]),
            "movement_distance_per_uav_transition": float(action_diag["movement_distance_per_uav_transition"]),
            "dt_switches": int(action_diag["dt_switches"]),
            "dt_host_switch_fraction": float(action_diag["dt_host_switch_fraction"]),
            "rehosted_entities_this_transition": float(action_diag["rehosted_entities_this_transition"]),
            "dt_switches_per_entity": action_diag["dt_switches_per_entity"].copy(),
            "executed_dt_assignment_index": int(action_diag["executed_dt_assignment_index"]),
            "sampled_dt_assignment_index": int(action_diag["sampled_dt_assignment_index"]),
            "reward": reward,
            "done": done,
        }
        self.transition_count += 1

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
        feasible_indices = self.feasible_dt_assignment_indices()
        dt_assignment_index = int(
            feasible_indices[
                self.base_env.rng.integers(low=0, high=len(feasible_indices))
            ]
        )

        action = {
            "uav_grid_indices": uav_grid_indices.astype(int),
            "dt_assignment_index": dt_assignment_index,
            "dt_hosts": self.base_env.enumerate_all_dt_assignments()[dt_assignment_index].astype(int),
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

    def feasible_dt_assignment_indices(self):
        return self.base_env.get_feasible_dt_assignment_indices()

    def current_dt_assignment_index(self):
        all_assignments = self.base_env.enumerate_all_dt_assignments()
        matches = np.all(all_assignments == self.base_env.dt_hosts[None, :], axis=1)
        assignment_indices = np.where(matches)[0]
        if len(assignment_indices) != 1:
            raise RuntimeError("Current DT-host assignment does not map to exactly one enumerated assignment.")
        return int(assignment_indices[0])

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
            num_entities=self.worker_env.base_env.E,
            lr=self.config["worker_lr"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_eps=self.config["clip_eps"],
            mask_worker_actions=True,
            worker_freshness_bias=self.config["worker_freshness_bias"],
            force_max_power=self.config["worker_force_max_power"],
            power_mode=self.config.get("worker_power_mode", "learned_beta"),
            continuous_power=self.config.get("worker_continuous_power", False),
            power_min=min(self.config["sensor_power_levels"]),
            power_max=max(self.config["sensor_power_levels"]),
            slot_duration=self.config["slot_duration"],
            bandwidth_access=self.config["bandwidth_access"],
            bandwidth_backhaul=self.config["bandwidth_backhaul"],
            noise_power=self.config["noise_power"],
            pathloss_ref=self.config["pathloss_ref"],
            pathloss_exp=self.config["pathloss_exp"],
            cpu_cycles_per_bit=self.config["cpu_cycles_per_bit"],
            cpu_rate=self.config["cpu_rate"],
            packet_size_max=self.config["packet_size_max"],
            area_size=self.config["area_size"],
            backhaul_power_max=self.config.get("backhaul_power_max", self.config["backhaul_power"]),
            backhaul_power_min=self.config.get("backhaul_power_min", self.config["backhaul_power"]),
            service_model=self.config.get("service_model", "abstract_same_step"),
        )

        if not self.worker_model_path.exists():
            raise FileNotFoundError(f"Worker model not found: {self.worker_model_path}")

        payload = torch.load(self.worker_model_path, map_location="cpu")
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if not metadata:
            raise RuntimeError(
                "Worker checkpoint is missing metadata required for manager-training compatibility checks."
            )
        required_matches = {
            "obs_spec_version": self.config.get("obs_spec_version"),
            "env_version": self.config.get("env_version"),
            "slot_duration": float(self.config.get("slot_duration")),
            "manager_horizon": int(self.config.get("manager_horizon")),
            "power_mode": self.config.get("worker_power_mode", "learned_beta"),
            "worker_context_mode": self.config.get("worker_context_mode", "random_feasible_context"),
            "scenario_distribution_version": self.config.get("scenario_distribution_version"),
            "hidden_dim": int(self.worker_agent.model.backbone[0].out_features),
        }
        for key, expected_value in required_matches.items():
            saved_value = metadata.get(key)
            if saved_value != expected_value:
                raise RuntimeError(
                    f"Incompatible worker checkpoint metadata for manager training: "
                    f"{key} saved={saved_value} current={expected_value}."
                )

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
        prev_positions = self.base_env.uav_positions.copy()
        prev_dt_hosts = self.base_env.dt_hosts.copy()

        if isinstance(action, dict):
            uav_grid_indices = action["uav_grid_indices"]
            dt_assignment_index = action.get("dt_assignment_index", None)
            dt_hosts = action.get("dt_hosts", None)
            backhaul_powers = action.get("backhaul_powers", None)
        else:
            if len(action) == 3:
                uav_grid_indices, dt_assignment_index, backhaul_powers = action
                dt_hosts = None
            else:
                uav_grid_indices, dt_assignment_index = action
                dt_hosts = None
                backhaul_powers = None

        uav_grid_indices = np.asarray(uav_grid_indices, dtype=int)

        if len(uav_grid_indices) != self.M:
            raise ValueError("Manager action must provide one grid index per UAV.")
        if np.any(uav_grid_indices < 0) or np.any(uav_grid_indices >= self.num_grid_points):
            raise ValueError("Manager grid index lies outside the configured UAV grid.")

        sampled_dt_assignment_index = -1
        if self.host_action_mode == "feasible_enum":
            if dt_assignment_index is None:
                if dt_hosts is None:
                    raise ValueError("Feasible-enum manager action must provide a DT-assignment index.")
                dt_hosts = np.asarray(dt_hosts, dtype=int)
                all_assignments = self.base_env.enumerate_all_dt_assignments()
                matches = np.all(all_assignments == dt_hosts[None, :], axis=1)
                found = np.where(matches)[0]
                if len(found) != 1:
                    raise ValueError("Provided dt_hosts does not correspond to a unique enumerated assignment.")
                dt_assignment_index = int(found[0])

            dt_assignment_index = int(dt_assignment_index)
            feasible_indices = self.feasible_dt_assignment_indices()
            if dt_assignment_index not in feasible_indices:
                raise ValueError("DT-assignment index is not feasible for the current storage setting.")
            dt_hosts = self.base_env.enumerate_all_dt_assignments()[dt_assignment_index].copy()
            sampled_dt_assignment_index = dt_assignment_index
        else:
            if dt_hosts is None:
                raise ValueError("Legacy-repair mode requires explicit dt_hosts.")
            dt_hosts = np.asarray(dt_hosts, dtype=int)
            if len(dt_hosts) != self.E:
                raise ValueError("Manager action must provide one DT host per entity.")
            if np.any(dt_hosts < 0) or np.any(dt_hosts >= self.M):
                raise ValueError("Manager DT host index lies outside the valid UAV range.")
            dt_hosts = self.base_env._repair_dt_storage(dt_hosts.copy())
            sampled_dt_assignment_index = -1

        self.base_env.uav_positions = self.grid_points[uav_grid_indices].copy()

        if self.optimize_backhaul_power:
            if backhaul_powers is None:
                backhaul_powers = np.ones(self.M) * self.config["backhaul_power"]

            backhaul_powers = np.asarray(backhaul_powers, dtype=float)
            if len(backhaul_powers) != self.M:
                raise ValueError("Manager action must provide one backhaul power per UAV.")
            if np.any(backhaul_powers < self.backhaul_power_min - 1e-9) or np.any(
                backhaul_powers > self.backhaul_power_max + 1e-9
            ):
                raise ValueError("Manager backhaul power lies outside configured physical bounds.")
            backhaul_powers = backhaul_powers.astype(np.float32)
        else:
            backhaul_powers = np.ones(self.M, dtype=np.float32) * self.config["backhaul_power"]

        self.base_env.apply_manager_context(
            uav_positions=self.base_env.uav_positions,
            dt_hosts=dt_hosts,
            backhaul_powers=backhaul_powers,
        )

        movement_distances = np.linalg.norm(self.base_env.uav_positions - prev_positions, axis=1)
        dt_switches_per_entity = (self.base_env.dt_hosts != prev_dt_hosts).astype(np.float32)
        raw_uav_switches = int(np.sum(movement_distances > 1e-9))
        raw_dt_switches = int(np.sum(dt_switches_per_entity))
        raw_movement_distance = float(np.sum(movement_distances))

        # Do not count the initial placement/hosting change after reset as a switch metric.
        if self.transition_count == 0:
            counted_uav_switches = 0
            counted_dt_switches = 0
            counted_movement_distance = 0.0
        else:
            counted_uav_switches = raw_uav_switches
            counted_dt_switches = raw_dt_switches
            counted_movement_distance = raw_movement_distance

        return {
            "uav_switches": int(counted_uav_switches),
            "uav_switch_fraction": float(counted_uav_switches / max(self.M, 1)),
            "changed_uavs_this_transition": float(counted_uav_switches),
            "movement_distance": float(counted_movement_distance),
            "movement_distance_per_uav_transition": float(
                counted_movement_distance / max(self.M, 1)
            ),
            "dt_switches": int(counted_dt_switches),
            "dt_host_switch_fraction": float(counted_dt_switches / max(self.E, 1)),
            "rehosted_entities_this_transition": float(counted_dt_switches),
            "dt_switches_per_entity": dt_switches_per_entity,
            "executed_dt_assignment_index": self.current_dt_assignment_index(),
            "sampled_dt_assignment_index": sampled_dt_assignment_index,
        }

    def _compute_reward(self, avg_window_aodt, avg_energy_per_uav, old_queues):
        normalized_aodt = avg_window_aodt / max(self.aoi_obs_norm, 1e-9)
        normalized_energy = avg_energy_per_uav / max(self.energy_budget, 1e-9)
        signed_violation = avg_energy_per_uav - self.energy_budget
        positive_violation = np.maximum(signed_violation, 0.0)
        updated_queues = np.maximum(0.0, old_queues + signed_violation)

        if self.manager_reward_mode == "legacy_queue_penalty":
            aodt_term = normalized_aodt
            energy_term = self.lyapunov_beta * float(
                np.sum(updated_queues * positive_violation)
            ) / max(self.energy_budget, 1e-9)
        elif self.manager_reward_mode == "queue_weighted_energy":
            normalized_queues = old_queues / max(self.energy_budget, 1e-9)
            aodt_term = self.manager_aodt_weight * normalized_aodt
            energy_term = self.manager_energy_weight * float(
                np.mean(normalized_queues * normalized_energy)
            )
        else:
            raise ValueError(f"Unknown manager reward mode: {self.manager_reward_mode}")

        reward = -float(aodt_term + energy_term)
        return reward, {
            "aodt_term": float(aodt_term),
            "energy_term": float(energy_term),
            "normalized_aodt": float(normalized_aodt),
            "normalized_energy_per_uav": normalized_energy.astype(np.float32).tolist(),
            "normalized_queue_per_uav": (old_queues / max(self.energy_budget, 1e-9)).astype(np.float32).tolist(),
            "raw_window_aodt": float(avg_window_aodt),
            "raw_avg_energy_per_uav": avg_energy_per_uav.astype(np.float32).tolist(),
            "energy_budget": float(self.energy_budget),
            "old_virtual_queues": old_queues.astype(np.float32).tolist(),
            "new_virtual_queues": updated_queues.astype(np.float32).tolist(),
            "signed_violation": signed_violation.astype(np.float32).tolist(),
            "positive_violation": positive_violation.astype(np.float32).tolist(),
            "reward_mode": self.manager_reward_mode,
            "total_reward": reward,
        }

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
        storage_scale = max(float(np.max(state["uav_storage_capacity"])), 1e-9)
        dt_storage = (state["dt_storage"] / storage_scale).astype(np.float32)
        uav_storage_capacity = (
            state["uav_storage_capacity"] / storage_scale
        ).astype(np.float32)

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
            dt_storage,
            uav_storage_capacity,
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
