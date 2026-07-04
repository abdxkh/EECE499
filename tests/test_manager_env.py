import unittest
from pathlib import Path

import numpy as np

from dt_uav_v2.agents.ppo import PPOAgent
from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.manager_env import ManagerEnv
from dt_uav_v2.envs.worker_env import WorkerEnv


def make_config():
    config = dict(CONFIG)
    config.update(
        {
            "seed": 13,
            "num_uavs": 2,
            "num_entities": 3,
            "num_sensors": 6,
            "area_size": 100.0,
            "episode_slots": 9,
            "manager_horizon": 3,
            "dt_storage_min": 40.0,
            "dt_storage_max": 40.0,
            "uav_storage_capacity": 80.0,
            "manager_host_action_mode": "feasible_enum",
            "manager_reward_mode": "queue_weighted_energy",
            "worker_power_mode": "fixed_max",
            "worker_continuous_power": True,
        }
    )
    return config


class ManagerEnvTests(unittest.TestCase):
    def test_feasible_assignment_enumeration_is_storage_feasible(self):
        env = ManagerEnv(config=make_config(), worker_policy="greedy")
        env.reset(seed=13)
        feasible_indices = env.feasible_dt_assignment_indices()
        self.assertGreater(len(feasible_indices), 0)
        all_assignments = env.base_env.enumerate_all_dt_assignments()
        for index in feasible_indices:
            assignment = all_assignments[index]
            used = env.base_env.compute_storage_used(assignment)
            self.assertTrue(np.all(used <= env.base_env.uav_storage_capacity + 1e-9))

    def test_sampled_and_executed_manager_assignment_match(self):
        env = ManagerEnv(config=make_config(), worker_policy="greedy")
        env.reset(seed=13)
        action = env.sample_random_manager_action()
        _, _, _, info = env.step(action)
        self.assertEqual(
            int(info["sampled_dt_assignment_index"]),
            int(info["executed_dt_assignment_index"]),
        )

    def test_virtual_queue_update_and_reward_terms(self):
        env = ManagerEnv(config=make_config(), worker_policy="greedy")
        env.reset(seed=13)
        env.virtual_queues = np.asarray([0.5, 0.25], dtype=np.float32)
        action = env.sample_random_manager_action()
        _, _, _, info = env.step(action)

        expected_updated = np.maximum(
            0.0,
            info["old_virtual_queues"] + info["energy_violation"],
        )
        np.testing.assert_allclose(info["virtual_queues"], expected_updated, atol=1e-6)

        normalized_energy = info["avg_energy_per_uav"] / env.energy_budget
        normalized_queues = info["old_virtual_queues"] / env.energy_budget
        expected_energy_term = float(np.mean(normalized_queues * normalized_energy))
        self.assertAlmostEqual(
            info["reward_terms"]["energy_term"],
            expected_energy_term,
            places=5,
        )

    def test_movement_and_dt_switch_diagnostics(self):
        env = ManagerEnv(config=make_config(), worker_policy="greedy")
        env.reset(seed=13)
        action = env.sample_random_manager_action()
        action["uav_grid_indices"] = np.asarray([0, env.num_grid_points - 1], dtype=int)
        feasible_indices = env.feasible_dt_assignment_indices()
        action["dt_assignment_index"] = int(feasible_indices[-1])
        action["dt_hosts"] = env.base_env.enumerate_all_dt_assignments()[action["dt_assignment_index"]]
        _, _, _, info = env.step(action)

        self.assertEqual(info["uav_switches"], 0)
        self.assertEqual(info["dt_switches"], 0)
        self.assertEqual(info["uav_switch_fraction"], 0.0)
        self.assertEqual(info["dt_host_switch_fraction"], 0.0)

        _, _, _, info_second = env.step(action)
        self.assertGreaterEqual(info_second["uav_switch_fraction"], 0.0)
        self.assertLessEqual(info_second["uav_switch_fraction"], 1.0)
        self.assertGreaterEqual(info_second["dt_host_switch_fraction"], 0.0)
        self.assertLessEqual(info_second["dt_host_switch_fraction"], 1.0)
        self.assertAlmostEqual(
            info_second["uav_switch_fraction"] * env.M,
            info_second["uav_switches"],
            places=6,
        )
        self.assertAlmostEqual(
            info_second["dt_host_switch_fraction"] * env.E,
            info_second["dt_switches"],
            places=6,
        )

    def test_manager_rejects_incompatible_worker_checkpoint_metadata(self):
        config = make_config()
        worker_env = WorkerEnv(config=config)
        worker_env.reset(seed=13)
        agent = PPOAgent(
            obs_dim=worker_env.obs_dim,
            num_uavs=worker_env.num_uavs,
            num_sensors=worker_env.num_sensors,
            num_power_levels=worker_env.num_power_levels,
            num_entities=worker_env.base_env.E,
            mask_worker_actions=True,
            power_mode=config["worker_power_mode"],
            continuous_power=True,
            power_min=min(worker_env.base_env.sensor_power_levels),
            power_max=max(worker_env.base_env.sensor_power_levels),
            slot_duration=worker_env.base_env.slot_duration,
            bandwidth_access=worker_env.base_env.B_access,
            bandwidth_backhaul=worker_env.base_env.B_backhaul,
            noise_power=worker_env.base_env.noise_power,
            pathloss_ref=worker_env.base_env.pathloss_ref,
            pathloss_exp=worker_env.base_env.pathloss_exp,
            cpu_cycles_per_bit=worker_env.base_env.cpu_cycles_per_bit,
            cpu_rate=worker_env.base_env.cpu_rate,
            packet_size_max=worker_env.packet_size_max,
            area_size=worker_env.area_size,
            backhaul_power_max=worker_env.config["backhaul_power_max"],
            backhaul_power_min=worker_env.config["backhaul_power_min"],
            service_model=worker_env.config.get("service_model", "abstract_same_step"),
        )
        tmp_dir = Path("outputs/test_tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / "worker_bad_metadata.pt"
        try:
            agent.save_checkpoint(
                path,
                extra_metadata={
                    "obs_spec_version": config["obs_spec_version"],
                    "env_version": config["env_version"],
                    "slot_duration": 99.0,
                    "manager_horizon": config["manager_horizon"],
                    "worker_context_mode": config["worker_context_mode"],
                    "scenario_distribution_version": config["scenario_distribution_version"],
                },
            )
            with self.assertRaises(RuntimeError):
                ManagerEnv(config=config, worker_policy="ppo", worker_model_path=path).reset(seed=13)
        finally:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    unittest.main()
