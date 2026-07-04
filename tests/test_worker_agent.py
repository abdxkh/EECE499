import unittest
from pathlib import Path

import numpy as np
import torch

from dt_uav_v2.agents.ppo import PPOAgent
from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.base_env import BaseUAVAoDTEnv
from dt_uav_v2.envs.worker_env import WorkerEnv


def make_config():
    config = dict(CONFIG)
    config.update(
        {
            "seed": 11,
            "num_uavs": 2,
            "num_entities": 2,
            "num_sensors": 4,
            "area_size": 100.0,
            "episode_slots": 5,
            "manager_horizon": 2,
            "manager_host_action_mode": "feasible_enum",
            "worker_continuous_power": True,
            "worker_power_mode": "learned_beta",
            "worker_reward_mode": "current",
        }
    )
    return config


def make_worker_agent(env, power_mode="learned_beta"):
    return PPOAgent(
        obs_dim=env.obs_dim,
        num_uavs=env.num_uavs,
        num_sensors=env.num_sensors,
        num_power_levels=env.num_power_levels,
        num_entities=env.base_env.E,
        mask_worker_actions=True,
        power_mode=power_mode,
        continuous_power=True,
        power_min=min(env.base_env.sensor_power_levels),
        power_max=max(env.base_env.sensor_power_levels),
        slot_duration=env.base_env.slot_duration,
        bandwidth_access=env.base_env.B_access,
        bandwidth_backhaul=env.base_env.B_backhaul,
        noise_power=env.base_env.noise_power,
        pathloss_ref=env.base_env.pathloss_ref,
        pathloss_exp=env.base_env.pathloss_exp,
        cpu_cycles_per_bit=env.base_env.cpu_cycles_per_bit,
        cpu_rate=env.base_env.cpu_rate,
        packet_size_max=env.packet_size_max,
        area_size=env.area_size,
        backhaul_power_max=env.config["backhaul_power_max"],
        backhaul_power_min=env.config["backhaul_power_min"],
        service_model=env.config.get("service_model", "abstract_same_step"),
    )


class WorkerAgentTests(unittest.TestCase):
    def test_current_worker_reward_mode(self):
        config = make_config()
        env = WorkerEnv(config=config)
        env.reset(seed=11)

        info = {
            "invalid_count": 2,
            "wasted_count": 1,
            "avg_aodt": 5.0,
        }

        reward = env._compute_reward(info, prev_avg_aodt=6.0)
        expected = (2.0 * (6.0 - 5.0) / 10.0) - (5.0 / 10.0) - (0.05 * 2 / 2) - (0.01 * 1 / 2)
        self.assertAlmostEqual(reward, expected, places=6)

    def test_fixed_max_uses_no_power_head(self):
        config = make_config()
        config["worker_power_mode"] = "fixed_max"
        env = WorkerEnv(config=config)
        env.reset(seed=11)
        agent = make_worker_agent(env, power_mode="fixed_max")
        self.assertIsNone(agent.model.power_head)

    def test_duplicate_sensor_exclusion(self):
        env = WorkerEnv(config=make_config())
        obs = env.reset(seed=11)
        env.base_env.Q[:] = 0.0
        env.base_env.Q[0] = 1.0
        obs = env._state_to_obs(env.base_env.get_basic_state())

        agent = make_worker_agent(env, power_mode="learned_beta")

        for _ in range(50):
            action, _, _, _ = agent.select_action(obs, deterministic=False)
            chosen = [sensor_id for sensor_id, _ in action if sensor_id != -1]
            self.assertEqual(len(chosen), len(set(chosen)))
            self.assertTrue(all(sensor_id == 0 for sensor_id in chosen))

    def test_idle_power_does_not_change_likelihood(self):
        env = WorkerEnv(config=make_config())
        env.reset(seed=11)
        env.base_env.Q[:] = 0.0
        env.base_env.Q[0] = 1.0
        obs = env._state_to_obs(env.base_env.get_basic_state())

        agent = make_worker_agent(env, power_mode="learned_beta")

        obs_batch = torch.as_tensor(obs, dtype=torch.float32).view(1, -1)
        actions_a = torch.tensor([[[0.0, 0.6], [float(env.num_sensors), 0.1]]], dtype=torch.float32)
        actions_b = torch.tensor([[[0.0, 0.6], [float(env.num_sensors), 0.9]]], dtype=torch.float32)
        log_prob_a, _, _ = agent._evaluate_actions(obs_batch, actions_a)
        log_prob_b, _, _ = agent._evaluate_actions(obs_batch, actions_b)
        self.assertAlmostEqual(float(log_prob_a.item()), float(log_prob_b.item()), places=6)

    def test_fixed_power_modes(self):
        config = make_config()
        for mode, expected in [("fixed_max", 0.2), ("fixed_mid", 0.125)]:
            config["worker_power_mode"] = mode
            env = WorkerEnv(config=config)
            obs = env.reset(seed=11)
            agent = make_worker_agent(env, power_mode=mode)
            action, _, _, _ = agent.select_action(obs, deterministic=True)
            selected = [power for sensor_id, power in action if sensor_id != -1]
            self.assertTrue(all(abs(power - expected) < 1e-6 for power in selected))

    def test_checkpoint_incompatibility_is_explicit(self):
        env = WorkerEnv(config=make_config())
        env.reset(seed=11)
        agent = make_worker_agent(env, power_mode="learned_beta")

        tmp_dir = Path("outputs/test_tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / "worker_checkpoint_incompatibility.pt"
        try:
            agent.save_checkpoint(path)

            incompatible_agent = PPOAgent(
                obs_dim=env.obs_dim + 1,
                num_uavs=env.num_uavs,
                num_sensors=env.num_sensors,
                num_power_levels=env.num_power_levels,
                num_entities=env.base_env.E,
                mask_worker_actions=True,
                power_mode="learned_beta",
                continuous_power=True,
                power_min=min(env.base_env.sensor_power_levels),
                power_max=max(env.base_env.sensor_power_levels),
                slot_duration=env.base_env.slot_duration,
                bandwidth_access=env.base_env.B_access,
                bandwidth_backhaul=env.base_env.B_backhaul,
                noise_power=env.base_env.noise_power,
                pathloss_ref=env.base_env.pathloss_ref,
                pathloss_exp=env.base_env.pathloss_exp,
                cpu_cycles_per_bit=env.base_env.cpu_cycles_per_bit,
                cpu_rate=env.base_env.cpu_rate,
                packet_size_max=env.packet_size_max,
                area_size=env.area_size,
                backhaul_power_max=env.config["backhaul_power_max"],
                backhaul_power_min=env.config["backhaul_power_min"],
                service_model=env.config.get("service_model", "abstract_same_step"),
            )
            with self.assertRaises(RuntimeError):
                incompatible_agent.load_checkpoint(path)
        finally:
            if path.exists():
                path.unlink()

    def test_delay_boundary_masking(self):
        config = make_config()
        config["worker_power_mode"] = "fixed_max"
        config["service_model"] = "require_within_slot"
        base_env = BaseUAVAoDTEnv(config=config)
        scenario = base_env.make_scenario_snapshot(seed=21)
        worker_env = WorkerEnv(config=config, base_env=base_env)
        worker_env.reset(scenario=scenario)

        base_env.sensor_positions = np.asarray([[0.0, 0.0], [90.0, 90.0], [95.0, 95.0], [98.0, 98.0]], dtype=np.float32)
        base_env.uav_positions = np.asarray([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
        base_env.sensor_entity = np.asarray([0, 1, 0, 1], dtype=int)
        base_env.dt_hosts = np.asarray([0, 1], dtype=int)
        base_env.Q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        base_env.W = np.asarray([8000.0, 0.0, 0.0, 0.0], dtype=np.float32)
        base_env.U = np.zeros(base_env.I, dtype=np.float32)
        obs = worker_env._state_to_obs(base_env.get_basic_state())
        power = max(base_env.sensor_power_levels)
        delay = base_env.W[0] / base_env.uplink_rate(0, 0, power) + base_env.processing_delay(0)

        def build_agent(slot_duration):
            return PPOAgent(
                obs_dim=worker_env.obs_dim or len(obs),
                num_uavs=worker_env.num_uavs,
                num_sensors=worker_env.num_sensors,
                num_power_levels=worker_env.num_power_levels,
                num_entities=worker_env.base_env.E,
                mask_worker_actions=True,
                power_mode="fixed_max",
                continuous_power=True,
                power_min=min(base_env.sensor_power_levels),
                power_max=max(base_env.sensor_power_levels),
                slot_duration=slot_duration,
                bandwidth_access=base_env.B_access,
                bandwidth_backhaul=base_env.B_backhaul,
                noise_power=base_env.noise_power,
                pathloss_ref=base_env.pathloss_ref,
                pathloss_exp=base_env.pathloss_exp,
                cpu_cycles_per_bit=base_env.cpu_cycles_per_bit,
                cpu_rate=base_env.cpu_rate,
                packet_size_max=worker_env.packet_size_max,
                area_size=worker_env.area_size,
                backhaul_power_max=worker_env.config["backhaul_power_max"],
                backhaul_power_min=worker_env.config["backhaul_power_min"],
                service_model="require_within_slot",
            )

        below = build_agent(float(delay) - 1e-5)
        exact = build_agent(float(delay))
        above = build_agent(float(delay) + 1e-5)
        selected = torch.zeros(worker_env.num_sensors, dtype=torch.bool, device=below.device)
        power_action = torch.tensor(1.0, dtype=torch.float32, device=below.device)

        mask_below = below._worker_sensor_mask(torch.as_tensor(obs, dtype=torch.float32), selected, 0, power_action)
        mask_exact = exact._worker_sensor_mask(torch.as_tensor(obs, dtype=torch.float32), selected, 0, power_action)
        mask_above = above._worker_sensor_mask(torch.as_tensor(obs, dtype=torch.float32), selected, 0, power_action)

        self.assertFalse(bool(mask_below[0].item()))
        self.assertTrue(bool(mask_exact[0].item()))
        self.assertTrue(bool(mask_above[0].item()))
        self.assertTrue(bool(mask_below[worker_env.num_sensors].item()))


if __name__ == "__main__":
    unittest.main()
