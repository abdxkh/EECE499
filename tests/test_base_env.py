import unittest

import numpy as np

from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.base_env import BaseUAVAoDTEnv


def make_small_config():
    config = dict(CONFIG)
    config.update(
        {
            "seed": 7,
            "num_uavs": 2,
            "num_entities": 2,
            "num_sensors": 4,
            "area_size": 100.0,
            "episode_slots": 6,
            "manager_horizon": 2,
            "arrival_prob": 1.0,
            "packet_size_min": 8000.0,
            "packet_size_max": 8000.0,
            "dt_storage_min": 40.0,
            "dt_storage_max": 40.0,
            "uav_storage_capacity": 100.0,
            "manager_host_action_mode": "feasible_enum",
        }
    )
    return config


def set_manual_state(env):
    env.sensor_positions = np.asarray(
        [[10.0, 10.0], [90.0, 10.0], [20.0, 20.0], [80.0, 20.0]],
        dtype=np.float32,
    )
    env.uav_positions = np.asarray([[0.0, 0.0], [100.0, 0.0]], dtype=np.float32)
    env.sensor_entity = np.asarray([0, 1, 0, 1], dtype=int)
    env.dt_hosts = np.asarray([0, 1], dtype=int)
    env.dt_storage = np.asarray([40.0, 40.0], dtype=np.float32)
    env.uav_storage_capacity = np.asarray([100.0, 100.0], dtype=np.float32)
    env.Q = np.ones(env.I, dtype=np.float32)
    env.W = np.ones(env.I, dtype=np.float32) * 8000.0
    env.U = np.zeros(env.I, dtype=np.float32)
    env.sensor_aoi = np.zeros(env.I, dtype=np.float32)
    env.entity_aodt = np.zeros(env.E, dtype=np.float32)
    env.last_backhaul_energy = np.zeros(env.M, dtype=np.float32)
    env.backhaul_powers = np.ones(env.M, dtype=np.float32) * env.backhaul_power
    env.update_entity_aodt()


class BaseEnvTests(unittest.TestCase):
    def test_buffer_overwrite_and_waiting_time(self):
        env = BaseUAVAoDTEnv(config=make_small_config())
        env.reset(seed=7)
        env.Q = np.asarray([1, 1, 0, 1], dtype=np.float32)
        env.W = np.asarray([10.0, 20.0, 0.0, 30.0], dtype=np.float32)
        env.U = np.asarray([2.0, 4.0, 0.0, 1.0], dtype=np.float32)

        arrivals = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        packet_sizes = np.asarray([99.0, 0.0, 0.0, 0.0], dtype=np.float32)
        served_prev = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        env.update_buffers(arrivals, packet_sizes, served_prev)

        np.testing.assert_allclose(env.Q, [1.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(env.W, [99.0, 0.0, 0.0, 30.0])
        np.testing.assert_allclose(env.U, [0.0, 0.0, 0.0, 2.0])

    def test_direct_and_cross_upload_energy(self):
        env = BaseUAVAoDTEnv(config=make_small_config())
        env.reset(seed=7)
        set_manual_state(env)

        _, info = env.step_worker([(0, 0.2), (1, 0.2)])
        self.assertAlmostEqual(float(info["backhaul_energy"][0]), 0.0, places=7)
        self.assertAlmostEqual(float(info["backhaul_energy"][1]), 0.0, places=7)

        env.reset(seed=7)
        set_manual_state(env)
        _, info = env.step_worker([(1, 0.2), (0, 0.2)])
        self.assertGreater(float(np.sum(info["backhaul_energy"])), 0.0)

    def test_entity_aodt_is_max_sensor_aoi(self):
        env = BaseUAVAoDTEnv(config=make_small_config())
        env.reset(seed=7)
        env.sensor_entity = np.asarray([0, 1, 0, 1], dtype=int)
        env.sensor_aoi = np.asarray([1.0, 4.0, 3.0, 2.0], dtype=np.float32)
        env.update_entity_aodt()
        np.testing.assert_allclose(env.entity_aodt, [3.0, 4.0])

    def test_same_scenario_replay_is_exact(self):
        config = make_small_config()
        env1 = BaseUAVAoDTEnv(config=config)
        scenario = env1.make_scenario_snapshot(seed=101)
        env2 = BaseUAVAoDTEnv(config=config)
        env1.reset(scenario=scenario)
        env2.reset(scenario=scenario)

        actions = [
            [(0, 0.2), (1, 0.2)],
            [(2, 0.2), (3, 0.2)],
            [(-1, 0.0), (1, 0.2)],
        ]

        for action in actions:
            _, info1 = env1.step_worker(action)
            _, info2 = env2.step_worker(action)
            np.testing.assert_allclose(info1["total_delay"], info2["total_delay"])
            np.testing.assert_allclose(info1["entity_aodt"], info2["entity_aodt"])
            np.testing.assert_allclose(info1["backhaul_energy"], info2["backhaul_energy"])

    def test_require_within_slot_keeps_packet_pending_on_overflow(self):
        config = make_small_config()
        config["slot_duration"] = 0.01
        config["service_model"] = "require_within_slot"
        env = BaseUAVAoDTEnv(config=config)
        env.reset(seed=7)
        set_manual_state(env)

        _, info = env.step_worker([(1, 0.05), (-1, 0.0)])

        self.assertEqual(float(info["attempted"][1]), 1.0)
        self.assertEqual(float(info["completed"][1]), 0.0)
        self.assertGreater(float(info["total_delay"][1]), config["slot_duration"])
        self.assertEqual(float(env.Q[1]), 1.0)

    def test_feasible_assignment_hash_is_deterministic(self):
        config = make_small_config()
        env_a = BaseUAVAoDTEnv(config=config)
        env_b = BaseUAVAoDTEnv(config=config)
        scenario = env_a.make_scenario_snapshot(seed=33)
        env_a.reset(scenario=scenario)
        env_b.reset(scenario=scenario)
        self.assertEqual(
            env_a.feasible_assignment_hash(),
            env_b.feasible_assignment_hash(),
        )


if __name__ == "__main__":
    unittest.main()
