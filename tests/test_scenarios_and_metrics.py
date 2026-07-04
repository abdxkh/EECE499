import unittest

import numpy as np

from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.base_env import BaseUAVAoDTEnv
from dt_uav_v2.utils.metrics import summarize_delays, summarize_manager_switching
from dt_uav_v2.utils.scenarios import make_scenario_suite, sample_worker_context, scenario_suite_metadata


def make_config():
    config = dict(CONFIG)
    config.update(
        {
            "seed": 17,
            "num_uavs": 3,
            "num_entities": 5,
            "num_sensors": 15,
            "episode_slots": 12,
            "manager_horizon": 3,
            "manager_host_action_mode": "feasible_enum",
        }
    )
    return config


class ScenarioAndMetricTests(unittest.TestCase):
    def test_scenario_suite_is_replayable_and_indexed(self):
        config = make_config()
        suite = make_scenario_suite(config=config, count=3, split="validation")
        metadata = scenario_suite_metadata(suite)
        self.assertEqual(metadata["count"], 3)
        self.assertEqual(metadata["split"], "validation")
        self.assertEqual(len(set(metadata["scenario_seeds"])), 3)

    def test_worker_contexts_are_valid(self):
        config = make_config()
        env = BaseUAVAoDTEnv(config=config)
        scenario = env.make_scenario_snapshot(seed=17)
        env.reset(scenario=scenario)

        for mode in ["fixed_context", "random_feasible_context", "heuristic_context"]:
            context = sample_worker_context(env, mode=mode, rng=np.random.default_rng(123), config=config)
            self.assertEqual(context["uav_positions"].shape, (env.M, 2))
            self.assertEqual(context["dt_hosts"].shape, (env.E,))
            self.assertEqual(context["backhaul_powers"].shape, (env.M,))
            self.assertTrue(np.all(context["backhaul_powers"] >= config["backhaul_power_min"] - 1e-9))
            self.assertTrue(np.all(context["backhaul_powers"] <= config["backhaul_power_max"] + 1e-9))
            self.assertTrue(
                np.all(env.compute_storage_used(context["dt_hosts"]) <= env.uav_storage_capacity + 1e-9)
            )

    def test_delay_summary(self):
        summary = summarize_delays([0.2, 0.4, 1.2], slot_duration=1.0)
        self.assertEqual(summary["count"], 3)
        self.assertAlmostEqual(summary["fraction_over_slot"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(summary["max_delay"], 1.2, places=6)

    def test_switch_summary(self):
        summary = summarize_manager_switching(
            manager_transitions=4,
            num_uavs=2,
            num_entities=3,
            uav_switches=[1, 0, 2, 1],
            movement_distances=[3.0, 0.0, 4.0, 1.0],
            dt_switches=[2, 1, 0, 1],
        )
        self.assertEqual(summary["manager_transitions"], 4)
        self.assertEqual(summary["raw_uav_position_change_count"], 4)
        self.assertEqual(summary["raw_dt_host_change_count"], 4)
        self.assertAlmostEqual(summary["uav_switch_fraction"], 0.5, places=6)
        self.assertAlmostEqual(summary["dt_host_switch_fraction"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(summary["total_grid_movement_distance"], 8.0, places=6)


if __name__ == "__main__":
    unittest.main()
