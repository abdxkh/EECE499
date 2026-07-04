import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dt_uav_v2.agents.manager_agent import ManagerPPOAgent
from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.manager_env import ManagerEnv
from dt_uav_v2.utils.metrics import summarize_delays, summarize_manager_switching
from dt_uav_v2.utils.scenarios import (
    make_scenario_aware_static_action,
    make_scenario_suite,
    nearest_grid_indices,
    scenario_suite_metadata,
)


def make_manager_agent(env, config, model_path):
    agent = ManagerPPOAgent(
        obs_dim=env.obs_dim,
        num_uavs=env.M,
        num_entities=env.E,
        num_grid_points=env.num_grid_points,
        num_host_actions=env.total_dt_assignments,
        lr=config["manager_lr"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_eps=config["clip_eps"],
        optimize_backhaul_power=config.get("optimize_backhaul_power", False),
        backhaul_power_min=config.get("backhaul_power_min", config["backhaul_power"]),
        backhaul_power_max=config.get("backhaul_power_max", config["backhaul_power"]),
        host_action_mode=config.get("manager_host_action_mode", "feasible_enum"),
    )
    agent.load(model_path)
    return agent


def make_fixed_initial_per_scenario_action(env):
    return {
        "uav_grid_indices": nearest_grid_indices(env.grid_points, env.base_env.uav_positions),
        "dt_assignment_index": env.current_dt_assignment_index(),
        "dt_hosts": env.base_env.dt_hosts.copy().astype(int),
        "backhaul_powers": env.base_env.backhaul_powers.copy().astype(np.float32),
    }


def make_fixed_global_action(env):
    if env.M == 3:
        uav_grid_indices = np.asarray([0, env.grid_size - 1, env.num_grid_points - 1], dtype=int)
    else:
        uav_grid_indices = np.arange(env.M, dtype=int) % env.num_grid_points
    dt_hosts = np.asarray([entity_id % env.M for entity_id in range(env.E)], dtype=int)
    backhaul_powers = np.ones(env.M, dtype=np.float32) * env.config["backhaul_power"]
    return {
        "uav_grid_indices": uav_grid_indices,
        "dt_assignment_index": None,
        "dt_hosts": dt_hosts,
        "backhaul_powers": backhaul_powers,
    }


def sample_random_manager_action(env, rng):
    feasible_indices = env.feasible_dt_assignment_indices()
    dt_assignment_index = int(feasible_indices[rng.integers(low=0, high=len(feasible_indices))])
    action = {
        "uav_grid_indices": rng.integers(low=0, high=env.num_grid_points, size=env.M).astype(int),
        "dt_assignment_index": dt_assignment_index,
        "dt_hosts": env.base_env.enumerate_all_dt_assignments()[dt_assignment_index].astype(int),
    }
    if env.optimize_backhaul_power:
        action["backhaul_powers"] = rng.uniform(
            low=env.backhaul_power_min,
            high=env.backhaul_power_max,
            size=env.M,
        ).astype(np.float32)
    return action


def run_manager_episode(env, policy_name, scenario, config, agent=None, rng=None):
    obs = env.reset(scenario=scenario)
    done = False

    fixed_initial_action = None
    fixed_global_action = None
    static_action = None
    if policy_name == "fixed_initial_per_scenario":
        fixed_initial_action = make_fixed_initial_per_scenario_action(env)
    elif policy_name == "fixed_global":
        fixed_global_action = make_fixed_global_action(env)
    elif policy_name == "static_heuristic":
        static_action = make_scenario_aware_static_action(env.base_env, config=config)

    rewards = []
    window_aodt = []
    tail_aodt = []
    mean_energy = []
    max_energy = []
    violation_flags = []
    queue_values = []
    invalid = 0
    wasted = 0
    windows = 0
    delays = []
    uav_switches = []
    movement_distances = []
    dt_switches = []
    reward_aodt_terms = []
    reward_energy_terms = []
    queue_vectors = []
    terminal_queue = np.zeros(env.M, dtype=np.float32)
    violation_windows = []
    energy_gt_aodt = []
    transition_rows = []

    while not done:
        if policy_name == "ppo":
            action, _, _, _ = agent.select_action(obs, deterministic=True)
        elif policy_name == "random":
            action = sample_random_manager_action(env, rng)
        elif policy_name == "fixed_global":
            action = fixed_global_action
        elif policy_name == "fixed_initial_per_scenario":
            action = fixed_initial_action
        elif policy_name == "static_heuristic":
            action = static_action
        else:
            raise ValueError(f"Unknown manager policy: {policy_name}")

        obs, reward, done, info = env.step(action)
        energy = np.asarray(info["avg_energy_per_uav"], dtype=np.float32)

        rewards.append(reward)
        window_aodt.append(info["avg_window_aodt"])
        tail_aodt.append(info["tail_window_aodt"])
        mean_energy.append(float(np.mean(energy)))
        max_energy.append(float(np.max(energy)))
        violation_flags.append(float(np.any(energy > env.energy_budget)))
        queue_values.append(float(np.mean(info["virtual_queues"])))
        invalid += info["invalid_count"]
        wasted += info["wasted_count"]
        uav_switches.append(int(info["uav_switches"]))
        movement_distances.append(float(info["movement_distance"]))
        dt_switches.append(int(info["dt_switches"]))
        reward_aodt_terms.append(float(info["reward_terms"]["aodt_term"]))
        reward_energy_terms.append(float(info["reward_terms"]["energy_term"]))
        queue_vectors.append(np.asarray(info["virtual_queues"], dtype=np.float32))
        terminal_queue = np.asarray(info["virtual_queues"], dtype=np.float32)
        violation_windows.extend((energy > env.energy_budget).astype(np.float32).tolist())
        energy_gt_aodt.append(float(info["reward_terms"]["energy_term"] > info["reward_terms"]["aodt_term"]))
        delays.extend(info.get("worker_slot_delay_stats", []))
        transition_rows.append(
            {
                "policy": policy_name,
                "scenario_seed": int(scenario["scenario_seed"]),
                "transition_index": int(windows),
                "window_aodt": float(info["avg_window_aodt"]),
                "normalized_aodt_term": float(info["reward_terms"]["aodt_term"]),
                "avg_energy_per_uav": energy.astype(np.float32).tolist(),
                "energy_budget": float(env.energy_budget),
                "old_virtual_queues": np.asarray(info["old_virtual_queues"], dtype=np.float32).tolist(),
                "new_virtual_queues": np.asarray(info["virtual_queues"], dtype=np.float32).tolist(),
                "signed_violation": np.asarray(info["energy_violation"], dtype=np.float32).tolist(),
                "positive_violation": np.asarray(info["positive_violation"], dtype=np.float32).tolist(),
                "queue_weighted_energy_term": float(info["reward_terms"]["energy_term"]),
                "total_manager_reward": float(reward),
            }
        )
        windows += 1

    switching = summarize_manager_switching(
        manager_transitions=windows,
        num_uavs=env.M,
        num_entities=env.E,
        uav_switches=uav_switches,
        movement_distances=movement_distances,
        dt_switches=dt_switches,
    )
    delay_summary = summarize_delays(delays, slot_duration=env.base_env.slot_duration)

    result = {
        "avg_reward": float(np.mean(rewards)),
        "avg_aodt": float(np.mean(window_aodt)),
        "tail_aodt": float(np.mean(tail_aodt)),
        "mean_energy": float(np.mean(mean_energy)),
        "max_energy": float(np.mean(max_energy)),
        "violation_rate": float(np.mean(violation_flags)),
        "avg_queue": float(np.mean(queue_values)),
        "final_queue": float(queue_values[-1]) if queue_values else 0.0,
        "invalid": int(invalid),
        "wasted": int(wasted),
        "windows": int(windows),
        "manager_actions": int(windows),
        "mean_aodt_reward_term": float(np.mean(reward_aodt_terms)) if reward_aodt_terms else 0.0,
        "mean_energy_reward_term": float(np.mean(reward_energy_terms)) if reward_energy_terms else 0.0,
        "max_energy_reward_term": float(np.max(reward_energy_terms)) if reward_energy_terms else 0.0,
        "energy_term_gt_aodt_fraction": float(np.mean(energy_gt_aodt)) if energy_gt_aodt else 0.0,
        "terminal_queue_per_uav": terminal_queue.astype(np.float32).tolist(),
        "mean_queue_per_uav": (
            np.mean(np.asarray(queue_vectors, dtype=np.float32), axis=0).astype(np.float32).tolist()
            if queue_vectors
            else np.zeros(env.M, dtype=np.float32).tolist()
        ),
        "uav_window_violation_fraction": float(np.mean(violation_windows)) if violation_windows else 0.0,
        **switching,
        **delay_summary,
    }
    return result, transition_rows


def write_results(output_dir, scenario_metadata, per_episode_rows, per_window_rows, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "scenario_suite.json", "w", encoding="ascii") as fh:
        json.dump(scenario_metadata, fh, indent=2)
    with open(output_dir / "summary.json", "w", encoding="ascii") as fh:
        json.dump(summary, fh, indent=2)
    if per_episode_rows:
        fieldnames = list(per_episode_rows[0].keys())
        with open(output_dir / "per_episode.csv", "w", newline="", encoding="ascii") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_episode_rows)
    if per_window_rows:
        fieldnames = list(per_window_rows[0].keys())
        with open(output_dir / "per_window.csv", "w", newline="", encoding="ascii") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_window_rows)


def evaluate_manager(
    manager_model_path="outputs/models/manager_backhaul_final.pt",
    worker_model_path="outputs/models/worker_continuous_final.pt",
    episodes=10,
    policies=None,
    worker_policy="ppo",
    seed=None,
    randomize_scenarios=False,
    output_dir=None,
):
    config = dict(CONFIG)
    if seed is not None:
        config["seed"] = seed

    policies = policies or ["ppo", "random", "fixed_global", "fixed_initial_per_scenario", "static_heuristic"]
    manager_model_path = Path(manager_model_path)
    worker_model_path = Path(worker_model_path)

    env = ManagerEnv(
        config=config,
        worker_model_path=worker_model_path,
        worker_policy=worker_policy,
    )
    initial_scenario = make_scenario_suite(config=config, count=1, split="test")[0]
    obs = env.reset(scenario=initial_scenario)

    agent = None
    if "ppo" in policies:
        if not manager_model_path.exists():
            raise FileNotFoundError(f"Manager model not found: {manager_model_path}")
        agent = make_manager_agent(env, config, manager_model_path)

    if randomize_scenarios:
        scenarios = make_scenario_suite(config=config, count=episodes, split="test")
    else:
        scenarios = [make_scenario_suite(config=config, count=1, split="test")[0] for _ in range(episodes)]

    print("Manager evaluation")
    print("Episodes:", episodes)
    print("Policies:", ", ".join(policies))
    print("Manager model:", manager_model_path)
    print("Worker model:", worker_model_path)
    print("Worker policy:", worker_policy)
    print("Observation dim:", len(obs))
    print("Energy budget:", env.energy_budget)
    print("Randomize scenarios:", randomize_scenarios)
    print("Reward mode:", config.get("manager_reward_mode", "queue_weighted_energy"))
    print("Host action mode:", config.get("manager_host_action_mode", "feasible_enum"))
    print("-" * 46)
    print(f"{'policy':<26}{'reward':>10}{'aodt':>10}{'energy':>10}")
    print("-" * 46)

    all_results = {}
    per_episode_rows = []
    per_window_rows = []
    policy_seed_offsets = {
        "ppo": 0,
        "random": 1_000,
        "fixed_global": 2_000,
        "fixed_initial_per_scenario": 3_000,
        "static_heuristic": 4_000,
    }

    for policy_name in policies:
        episode_results = []
        policy_rng = np.random.default_rng(config["seed"] + policy_seed_offsets[policy_name])

        for episode_index, scenario in enumerate(scenarios):
            result, transition_rows = run_manager_episode(
                env=env,
                policy_name=policy_name,
                scenario=scenario,
                config=config,
                agent=agent,
                rng=policy_rng,
            )
            episode_results.append(result)
            per_window_rows.extend(transition_rows)
            per_episode_rows.append(
                {
                    "policy": policy_name,
                    "episode_index": episode_index,
                    "scenario_seed": int(scenario["scenario_seed"]),
                    **result,
                }
            )

        summary = {}
        for key in episode_results[0].keys():
            values = [result[key] for result in episode_results]
            if isinstance(values[0], list):
                summary[key] = np.mean(np.asarray(values, dtype=np.float32), axis=0).astype(np.float32).tolist()
            else:
                summary[key] = float(np.mean(values))
        all_results[policy_name] = summary

        print(
            f"{policy_name:<26}"
            f"{summary['avg_reward']:>10.4f}"
            f"{summary['avg_aodt']:>10.4f}"
            f"{summary['mean_energy']:>10.4f}"
        )

    print("-" * 46)

    if output_dir is not None:
        write_results(
            output_dir=Path(output_dir),
            scenario_metadata=scenario_suite_metadata(scenarios),
            per_episode_rows=per_episode_rows,
            per_window_rows=per_window_rows,
            summary=all_results,
        )

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate manager policies.")
    parser.add_argument("--manager-model-path", type=str, default="outputs/models/manager_backhaul_final.pt")
    parser.add_argument("--worker-model-path", type=str, default="outputs/models/worker_continuous_final.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["ppo", "random", "fixed_global", "fixed_initial_per_scenario", "static_heuristic"],
        choices=["ppo", "random", "fixed_global", "fixed_initial_per_scenario", "static_heuristic"],
    )
    parser.add_argument("--worker-policy", choices=["ppo", "greedy"], default="ppo")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--randomize-scenarios", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_manager(
        manager_model_path=args.manager_model_path,
        worker_model_path=args.worker_model_path,
        episodes=args.episodes,
        policies=args.policies,
        worker_policy=args.worker_policy,
        seed=args.seed,
        randomize_scenarios=args.randomize_scenarios,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
