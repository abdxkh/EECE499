import argparse
import copy
import csv
import json
import traceback
import time
from pathlib import Path

import numpy as np
import torch

from dt_uav_v2.agents.ppo import PPOAgent
from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.manager_env import ManagerEnv
from dt_uav_v2.envs.worker_env import WorkerEnv
from dt_uav_v2.evaluation.evaluate import make_manager_agent
from dt_uav_v2.training.train_manager import train_manager
from dt_uav_v2.training.train_worker import apply_worker_context, prepare_worker_config, train_worker
from dt_uav_v2.utils.metrics import summarize_delays, summarize_manager_switching
from dt_uav_v2.utils.scenarios import make_scenario_suite, sample_worker_context, scenario_suite_metadata


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as fh:
        json.dump(data, fh, indent=2)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json_if_exists(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="ascii") as fh:
        return json.load(fh)


def load_csv_rows_if_exists(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="ascii", newline="") as fh:
        reader = csv.DictReader(fh)
        return [{key: parse_csv_value(value) for key, value in row.items()} for row in reader]


def parse_csv_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except Exception:
        try:
            return float(text)
        except Exception:
            return text


def run_status_path(output_dir, kind, identifier):
    return output_dir / "status" / f"{kind}_{identifier}.json"


def write_run_status(output_dir, kind, identifier, payload):
    write_json(run_status_path(output_dir, kind, identifier), payload)


def worker_checkpoint_matches_config(metadata, config):
    expected_power_mode = config.get("worker_power_mode", "fixed_max")
    checks = [
        metadata.get("obs_spec_version") == config.get("obs_spec_version"),
        metadata.get("env_version") == config.get("env_version"),
        metadata.get("scenario_distribution_version") == config.get("scenario_distribution_version"),
        float(metadata.get("slot_duration", -1.0)) == float(config.get("slot_duration", -2.0)),
        int(metadata.get("manager_horizon", -1)) == int(config.get("manager_horizon", -2)),
        metadata.get("worker_power_mode") == expected_power_mode,
        metadata.get("worker_context_mode") == config.get("worker_context_mode", "random_feasible_context"),
        metadata.get("architecture_variant") == f"worker_{expected_power_mode}",
    ]
    return all(checks)


def manager_checkpoint_matches_config(metadata, config):
    checks = [
        metadata.get("obs_spec_version") == config.get("obs_spec_version"),
        metadata.get("env_version") == config.get("env_version"),
        metadata.get("scenario_distribution_version") == config.get("scenario_distribution_version"),
        float(metadata.get("slot_duration", -1.0)) == float(config.get("slot_duration", -2.0)),
        int(metadata.get("manager_horizon", -1)) == int(config.get("manager_horizon", -2)),
        metadata.get("worker_power_mode") == config.get("worker_power_mode", "fixed_max"),
        metadata.get("worker_context_mode") == config.get("worker_context_mode", "random_feasible_context"),
        metadata.get("architecture_variant") == "manager_feasible_enum",
    ]
    return all(checks)


def load_completed_worker_run(output_dir, seed, config):
    stem = f"worker_fixed_max_seed{seed}"
    model_path = output_dir / "models" / f"{stem}.pt"
    summary_path = output_dir / "validation" / f"{stem}.json"
    episode_path = output_dir / "validation" / f"{stem}.csv"
    transmission_path = output_dir / "validation" / f"{stem}_transmissions.csv"
    regret_path = output_dir / "validation" / f"{stem}_regret.csv"
    status = load_json_if_exists(run_status_path(output_dir, "worker", stem))
    if status is not None and status.get("status") != "completed":
        return None
    if not (model_path.exists() and summary_path.exists() and episode_path.exists() and transmission_path.exists() and regret_path.exists()):
        return None
    metadata = load_checkpoint_metadata(model_path)
    if not worker_checkpoint_matches_config(metadata, config):
        return None
    return {
        "seed": int(seed),
        "config": dict(config),
        "model_path": str(model_path),
        "history_path": str(output_dir / "history" / stem),
        "summary": load_json_if_exists(summary_path) or {},
        "episode_rows": load_csv_rows_if_exists(episode_path),
        "transmission_rows": load_csv_rows_if_exists(transmission_path),
        "regret_rows": load_csv_rows_if_exists(regret_path),
        "checkpoint_metadata": metadata,
        "status": "completed",
        "reused": True,
    }


def load_completed_manager_run(output_dir, worker_policy, rollout_size, lr, entropy_coef, seed, config):
    stem = f"manager_{worker_policy}_r{int(rollout_size)}_lr{float(lr):.0e}_e{float(entropy_coef):.3g}_seed{int(seed)}"
    model_path = output_dir / "models" / f"{stem}.pt"
    summary_path = output_dir / "validation" / f"{stem}.json"
    static_summary_path = output_dir / "validation" / f"{stem}_static.json"
    episode_path = output_dir / "validation" / f"{stem}.csv"
    transition_path = output_dir / "validation" / f"{stem}_transitions.csv"
    transmission_path = output_dir / "validation" / f"{stem}_transmissions.csv"
    static_episode_path = output_dir / "validation" / f"{stem}_static.csv"
    static_transition_path = output_dir / "validation" / f"{stem}_static_transitions.csv"
    static_transmission_path = output_dir / "validation" / f"{stem}_static_transmissions.csv"
    status = load_json_if_exists(run_status_path(output_dir, "manager", stem))
    if status is not None and status.get("status") != "completed":
        return None
    if not (
        model_path.exists()
        and summary_path.exists()
        and static_summary_path.exists()
        and episode_path.exists()
        and transition_path.exists()
        and transmission_path.exists()
        and static_episode_path.exists()
        and static_transition_path.exists()
        and static_transmission_path.exists()
    ):
        return None
    metadata = load_checkpoint_metadata(model_path)
    if not manager_checkpoint_matches_config(metadata, config):
        return None
    return {
        "seed": int(seed),
        "config": dict(config),
        "model_path": str(model_path),
        "history_path": str(output_dir / "history" / stem),
        "summary": load_json_if_exists(summary_path) or {},
        "static_summary": load_json_if_exists(static_summary_path) or {},
        "episode_rows": load_csv_rows_if_exists(episode_path),
        "transition_rows": load_csv_rows_if_exists(transition_path),
        "transmission_rows": load_csv_rows_if_exists(transmission_path),
        "static_episode_rows": load_csv_rows_if_exists(static_episode_path),
        "static_transition_rows": load_csv_rows_if_exists(static_transition_path),
        "static_transmission_rows": load_csv_rows_if_exists(static_transmission_path),
        "checkpoint_metadata": metadata,
        "status": "completed",
        "reused": True,
    }


def print_execution_plan(args, config):
    worker_runs = 0 if args.manager_only else len(args.worker_seeds)
    manager_runs = 0
    if not args.worker_only:
        stage_a = len(args.manager_rollout_sizes) * min(3, len(args.manager_seeds))
        stage_b = len(args.manager_lrs) * min(3, len(args.manager_seeds))
        stage_c = len(args.manager_entropy_coefs) * min(3, len(args.manager_seeds))
        finalists = min(2, len(args.manager_entropy_coefs)) * len(args.manager_seeds)
        greedy = len(args.manager_seeds)
        manager_runs = stage_a + stage_b + stage_c + finalists + greedy
    worker_steps_per_run = int(args.worker_episodes) * int(config["episode_slots"])
    manager_steps_per_run = int(args.manager_episodes) * int(np.ceil(config["episode_slots"] / config["manager_horizon"]))
    total_worker_steps = worker_runs * worker_steps_per_run
    total_manager_steps = manager_runs * manager_steps_per_run
    total_unique_seeds = len(args.worker_seeds) + len(args.manager_seeds)
    expected_dirs = [
        "worker",
        "manager_stage_a",
        "manager_stage_b",
        "manager_stage_c",
        "manager_final_candidate_0",
        "manager_final_candidate_1",
        "manager_greedy_worker",
        "validation",
        "history",
        "models",
        "status",
    ]
    print("Phase C2 execution plan")
    print(f"  output_dir: {args.output_dir}")
    print(f"  worker_runs: {worker_runs}")
    print(f"  manager_runs: {manager_runs}")
    print(f"  total_unique_seeds: {total_unique_seeds}")
    print(f"  approx_worker_env_steps: {total_worker_steps}")
    print(f"  approx_manager_env_steps: {total_manager_steps}")
    print(f"  approx_total_env_steps: {total_worker_steps + total_manager_steps}")
    print("  expected_output_dirs:")
    for entry in expected_dirs:
        print(f"    - {entry}")
    return {
        "output_dir": args.output_dir,
        "worker_runs": int(worker_runs),
        "manager_runs": int(manager_runs),
        "total_unique_seeds": int(total_unique_seeds),
        "approx_worker_env_steps": int(total_worker_steps),
        "approx_manager_env_steps": int(total_manager_steps),
        "approx_total_env_steps": int(total_worker_steps + total_manager_steps),
        "expected_output_dirs": expected_dirs,
    }


def build_phase_c2_config():
    config = prepare_worker_config(dict(CONFIG))
    config.update(
        {
            "slot_duration": 1.0,
            "manager_horizon": 5,
            "service_model": "require_within_slot",
            "worker_power_mode": "fixed_max",
            "worker_context_mode": "random_feasible_context",
            "manager_host_action_mode": "feasible_enum",
            "manager_reward_mode": "queue_weighted_energy",
            "manager_aodt_weight": 20.0,
            "manager_energy_weight": 1.0,
            "manager_entropy_coef": 0.01,
            "validation_worker_scenarios": 40,
            "validation_manager_scenarios": 40,
            "optimize_backhaul_power": True,
        }
    )
    return config


def load_checkpoint_metadata(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        return payload.get("metadata", {})
    return {}


def make_worker_agent(env, config, model_path):
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        num_uavs=env.num_uavs,
        num_sensors=env.num_sensors,
        num_power_levels=env.num_power_levels,
        num_entities=env.base_env.E,
        lr=config["worker_lr"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_eps=config["clip_eps"],
        mask_worker_actions=True,
        worker_freshness_bias=config["worker_freshness_bias"],
        power_mode=config.get("worker_power_mode", "fixed_max"),
        force_max_power=config["worker_force_max_power"],
        continuous_power=config.get("worker_continuous_power", True),
        power_min=min(config["sensor_power_levels"]),
        power_max=max(config["sensor_power_levels"]),
        slot_duration=config["slot_duration"],
        bandwidth_access=config["bandwidth_access"],
        bandwidth_backhaul=config["bandwidth_backhaul"],
        noise_power=config["noise_power"],
        pathloss_ref=config["pathloss_ref"],
        pathloss_exp=config["pathloss_exp"],
        cpu_cycles_per_bit=config["cpu_cycles_per_bit"],
        cpu_rate=config["cpu_rate"],
        packet_size_max=config["packet_size_max"],
        area_size=config["area_size"],
        backhaul_power_max=config.get("backhaul_power_max", config["backhaul_power"]),
        backhaul_power_min=config.get("backhaul_power_min", config["backhaul_power"]),
        service_model=config.get("service_model", "require_within_slot"),
    )
    agent.load(model_path)
    return agent


def estimate_worker_action_gain(env, action):
    cloned = copy.deepcopy(env)
    prev = cloned.base_env.average_aodt()
    _, _, _, info = cloned.step(action)
    return float(prev - float(info["avg_aodt"])), info


def evaluate_worker_policy(policy_name, config, model_path, scenarios, compare_same_state=False):
    env = WorkerEnv(config=config)
    agent = None
    if policy_name == "ppo":
        env.reset(scenario=scenarios[0])
        context = sample_worker_context(
            env.base_env,
            mode=config.get("worker_context_mode", "random_feasible_context"),
            rng=np.random.default_rng(int(scenarios[0]["scenario_seed"]) + 123),
            config=config,
        )
        apply_worker_context(env, context)
        agent = make_worker_agent(env, config, model_path)

    episode_rows = []
    transmission_rows = []
    regret_rows = []
    inference_time = []
    action_entropy = []
    same_sensor_matches = []
    ppo_gain_values = []
    greedy_gain_values = []
    regret_values = []
    feasible_action_counts = []
    no_feasible_counts = []

    for episode_index, scenario in enumerate(scenarios):
        env.reset(scenario=scenario)
        context = sample_worker_context(
            env.base_env,
            mode=config.get("worker_context_mode", "random_feasible_context"),
            rng=np.random.default_rng(int(scenario["scenario_seed"]) + 123),
            config=config,
        )
        obs = apply_worker_context(env, context)
        rng = np.random.default_rng(int(scenario["scenario_seed"]) + 999)
        done = False
        ep_rewards = []
        ep_aodt = []
        ep_delays = []
        ep_idle = 0
        ep_actions = 0
        ep_direct = 0
        ep_cross = 0
        ep_completions = 0
        ep_invalid = 0
        ep_wasted = 0
        ep_times = []
        ep_entropy = []
        ep_same = []
        ep_ppo_gain = []
        ep_greedy_gain = []
        ep_regret = []
        ep_feasible = []
        ep_no_feasible = []

        while not done:
            if policy_name == "ppo":
                t0 = time.perf_counter()
                action, action_indices, _, _ = agent.select_action(obs, deterministic=True)
                ep_times.append(time.perf_counter() - t0)
                if compare_same_state:
                    greedy_action = env.sample_max_age_reduction_action()
                    ppo_gain, _ = estimate_worker_action_gain(env, action)
                    greedy_gain, _ = estimate_worker_action_gain(env, greedy_action)
                    same = float(
                        np.mean([int(a[0]) == int(b[0]) for a, b in zip(action, greedy_action)])
                    )
                    ep_same.append(same)
                    ep_ppo_gain.append(ppo_gain)
                    ep_greedy_gain.append(greedy_gain)
                    ep_regret.append(greedy_gain - ppo_gain)
                    feasible_mask = agent.worker_feasible_real_mask(obs, deterministic=True)
                    ep_feasible.append(float(np.sum(feasible_mask)))
                    ep_no_feasible.append(float(np.sum(np.sum(feasible_mask, axis=1) == 0)))
                    action_tensor = torch.as_tensor(action_indices, dtype=torch.float32, device=agent.device).view(1, env.num_uavs, 2)
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).view(1, -1)
                    with torch.no_grad():
                        _, entropy, _ = agent._evaluate_actions(obs_tensor, action_tensor)
                    ep_entropy.append(float(entropy.item()))
                    regret_rows.append(
                        {
                            "policy": policy_name,
                            "episode_index": episode_index,
                            "scenario_seed": int(scenario["scenario_seed"]),
                            "state_index": len(ep_same) - 1,
                            "ppo_same_sensor_fraction": same,
                            "ppo_one_step_aodt_reduction": float(ppo_gain),
                            "greedy_one_step_aodt_reduction": float(greedy_gain),
                            "regret": float(greedy_gain - ppo_gain),
                            "action_entropy": float(entropy.item()),
                            "feasible_real_action_count": float(np.sum(feasible_mask)),
                            "no_feasible_real_sensor_count": float(np.sum(np.sum(feasible_mask, axis=1) == 0)),
                        }
                    )
            elif policy_name == "greedy":
                t0 = time.perf_counter()
                action = env.sample_max_age_reduction_action()
                ep_times.append(time.perf_counter() - t0)
            elif policy_name == "proximity":
                t0 = time.perf_counter()
                action = env.sample_proximity_greedy_action()
                ep_times.append(time.perf_counter() - t0)
            elif policy_name == "random":
                t0 = time.perf_counter()
                action = sample_valid_random_action(env, rng)
                ep_times.append(time.perf_counter() - t0)
            else:
                raise ValueError(f"Unknown worker policy: {policy_name}")

            for sensor_id, power in action:
                if sensor_id == -1:
                    ep_idle += 1
                ep_actions += 1

            obs, reward, done, info = env.step(action)
            ep_rewards.append(float(reward))
            ep_aodt.append(float(info["avg_aodt"]))
            ep_invalid += int(info["invalid_count"])
            ep_wasted += int(info["wasted_count"])
            for record in info.get("transmission_records", []):
                if record["completed"]:
                    ep_delays.append(float(record["total_delay_s"]))
                    if record["direct_upload"]:
                        ep_direct += 1
                    else:
                        ep_cross += 1
                    ep_completions += 1
                transmission_rows.append(
                    {
                        "policy": policy_name,
                        "episode_index": episode_index,
                        "scenario_seed": int(scenario["scenario_seed"]),
                        **record,
                    }
                )

        episode_rows.append(
            {
                "policy": policy_name,
                "episode_index": episode_index,
                "scenario_seed": int(scenario["scenario_seed"]),
                "avg_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
                "avg_aodt": float(np.mean(ep_aodt)) if ep_aodt else 0.0,
                "invalid": int(ep_invalid),
                "wasted": int(ep_wasted),
                "mean_delay": float(np.mean(ep_delays)) if ep_delays else 0.0,
                "p95_delay": float(np.percentile(ep_delays, 95)) if ep_delays else 0.0,
                "idle_rate": float(ep_idle / max(ep_actions, 1)),
                "direct_ratio": float(ep_direct / max(ep_completions, 1)),
                "cross_ratio": float(ep_cross / max(ep_completions, 1)),
                "completed_updates": int(ep_completions),
                "mean_action_time_s": float(np.mean(ep_times)) if ep_times else 0.0,
                "mean_entropy": float(np.mean(ep_entropy)) if ep_entropy else None,
                "mean_same_sensor_fraction": float(np.mean(ep_same)) if ep_same else None,
                "mean_ppo_gain": float(np.mean(ep_ppo_gain)) if ep_ppo_gain else None,
                "mean_greedy_gain": float(np.mean(ep_greedy_gain)) if ep_greedy_gain else None,
                "mean_regret": float(np.mean(ep_regret)) if ep_regret else None,
                "feasible_real_action_count": float(np.mean(ep_feasible)) if ep_feasible else None,
                "no_feasible_real_sensor_count": float(np.mean(ep_no_feasible)) if ep_no_feasible else None,
            }
        )
        if ep_times:
            inference_time.extend(ep_times)
            action_entropy.extend(ep_entropy)
            same_sensor_matches.extend(ep_same)
            ppo_gain_values.extend(ep_ppo_gain)
            greedy_gain_values.extend(ep_greedy_gain)
            regret_values.extend(ep_regret)
            feasible_action_counts.extend(ep_feasible)
            no_feasible_counts.extend(ep_no_feasible)

    completed_records = [row for row in transmission_rows if row["completed"]]
    direct_records = [row for row in completed_records if row["direct_upload"]]
    cross_records = [row for row in completed_records if row["cross_upload"]]

    summary = {
        "policy": policy_name,
        "avg_reward": float(np.mean([row["avg_reward"] for row in episode_rows])) if episode_rows else 0.0,
        "avg_aodt": float(np.mean([row["avg_aodt"] for row in episode_rows])) if episode_rows else 0.0,
        "idle_rate": float(np.mean([row["idle_rate"] for row in episode_rows])) if episode_rows else 0.0,
        "direct_ratio": float(np.mean([row["direct_ratio"] for row in episode_rows])) if episode_rows else 0.0,
        "cross_ratio": float(np.mean([row["cross_ratio"] for row in episode_rows])) if episode_rows else 0.0,
        "completed_updates": int(sum(row["completed_updates"] for row in episode_rows)),
        "inference_time_s_total": float(np.sum(inference_time)) if inference_time else 0.0,
        "inference_time_s_mean": float(np.mean(inference_time)) if inference_time else 0.0,
        "action_entropy_mean": float(np.mean(action_entropy)) if action_entropy else None,
        "same_sensor_fraction_mean": float(np.mean(same_sensor_matches)) if same_sensor_matches else None,
        "ppo_gain_mean": float(np.mean(ppo_gain_values)) if ppo_gain_values else None,
        "greedy_gain_mean": float(np.mean(greedy_gain_values)) if greedy_gain_values else None,
        "regret_mean": float(np.mean(regret_values)) if regret_values else None,
        "feasible_real_action_count_mean": float(np.mean(feasible_action_counts)) if feasible_action_counts else None,
        "no_feasible_real_sensor_count_mean": float(np.mean(no_feasible_counts)) if no_feasible_counts else None,
        "count": int(len(completed_records)),
        **summarize_delays([record["total_delay_s"] for record in completed_records], slot_duration=config["slot_duration"]),
        "completed_summary": summarize_delays([record["total_delay_s"] for record in completed_records], slot_duration=config["slot_duration"]),
        "direct_summary": summarize_delays([record["total_delay_s"] for record in direct_records], slot_duration=config["slot_duration"]),
        "cross_summary": summarize_delays([record["total_delay_s"] for record in cross_records], slot_duration=config["slot_duration"]),
    }
    summary["mean_action_time_s"] = float(np.mean(inference_time)) if inference_time else 0.0
    summary["action_entropy_mean"] = safe_mean(action_entropy)
    summary["same_sensor_fraction_mean"] = safe_mean(same_sensor_matches)
    summary["ppo_gain_mean"] = safe_mean(ppo_gain_values)
    summary["greedy_gain_mean"] = safe_mean(greedy_gain_values)
    summary["regret_mean"] = safe_mean(regret_values)
    summary["feasible_real_action_count_mean"] = safe_mean(feasible_action_counts)
    summary["no_feasible_real_sensor_count_mean"] = safe_mean(no_feasible_counts)
    return summary, episode_rows, transmission_rows, regret_rows


def evaluate_manager_policy(policy_name, env, scenarios, config, agent=None, rng_seed=0):
    episode_rows = []
    transition_rows = []
    transmission_rows = []
    inference_times = []
    policy_rng = np.random.default_rng(int(rng_seed))

    for episode_index, scenario in enumerate(scenarios):
        obs = env.reset(scenario=scenario)
        done = False
        fixed_initial_action = None
        fixed_global_action = None
        static_action = None
        if policy_name == "fixed_initial_per_scenario":
            fixed_initial_action = {
                "uav_grid_indices": env.base_env.uav_positions.copy(),
            }
            fixed_initial_action = {
                "uav_grid_indices": np.asarray(np.argmin(np.linalg.norm(env.grid_points[:, None, :] - env.base_env.uav_positions[None, :, :], axis=2), axis=0), dtype=int),
                "dt_assignment_index": env.current_dt_assignment_index(),
                "dt_hosts": env.base_env.dt_hosts.copy().astype(int),
                "backhaul_powers": env.base_env.backhaul_powers.copy().astype(np.float32),
            }
        elif policy_name == "fixed_global":
            if env.M == 3:
                uav_grid_indices = np.asarray([0, env.grid_size - 1, env.num_grid_points - 1], dtype=int)
            else:
                uav_grid_indices = np.arange(env.M, dtype=int) % env.num_grid_points
            fixed_global_action = {
                "uav_grid_indices": uav_grid_indices,
                "dt_assignment_index": None,
                "dt_hosts": np.asarray([entity_id % env.M for entity_id in range(env.E)], dtype=int),
                "backhaul_powers": np.ones(env.M, dtype=np.float32) * env.config["backhaul_power"],
            }
        elif policy_name == "static_heuristic":
            from dt_uav_v2.utils.scenarios import make_scenario_aware_static_action

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
        total_select_time = 0.0
        total_action_calls = 0
        total_worker_select_time = 0.0
        total_worker_idle = 0
        total_worker_actions = 0

        while not done:
            t0 = time.perf_counter()
            if policy_name == "ppo":
                action, _, _, _ = agent.select_action(obs, deterministic=True)
            elif policy_name == "random":
                action = env.sample_random_manager_action()
            elif policy_name == "fixed_global":
                action = fixed_global_action
            elif policy_name == "fixed_initial_per_scenario":
                action = fixed_initial_action
            elif policy_name == "static_heuristic":
                action = static_action
            else:
                raise ValueError(f"Unknown manager policy: {policy_name}")
            total_select_time += time.perf_counter() - t0
            total_action_calls += 1

            obs, reward, done, info = env.step(action)
            total_worker_select_time += float(info.get("worker_select_time_total", 0.0))
            total_worker_idle += int(info.get("worker_idle_count", 0))
            total_worker_actions += int(info.get("worker_action_count", 0))
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
            delays.extend(info.get("worker_slot_delay_stats", []))
            transition_rows.append(
                {
                    "policy": policy_name,
                    "episode_index": episode_index,
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
                    "uav_switch_fraction": float(info["uav_switch_fraction"]),
                    "dt_host_switch_fraction": float(info["dt_host_switch_fraction"]),
                    "movement_distance": float(info["movement_distance"]),
                }
            )
            for record in info.get("transmission_records", []):
                transmission_rows.append(
                    {
                        "policy": policy_name,
                        "episode_index": episode_index,
                        "scenario_seed": int(scenario["scenario_seed"]),
                        **record,
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
        episode_rows.append(
            {
                "policy": policy_name,
                "episode_index": episode_index,
                "scenario_seed": int(scenario["scenario_seed"]),
                "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
                "avg_aodt": float(np.mean(window_aodt)) if window_aodt else 0.0,
                "tail_aodt": float(np.mean(tail_aodt)) if tail_aodt else 0.0,
                "mean_energy": float(np.mean(mean_energy)) if mean_energy else 0.0,
                "max_energy": float(np.mean(max_energy)) if max_energy else 0.0,
                "violation_rate": float(np.mean(violation_flags)) if violation_flags else 0.0,
                "avg_queue": float(np.mean(queue_values)) if queue_values else 0.0,
                "final_queue": float(queue_values[-1]) if queue_values else 0.0,
                "invalid": int(invalid),
                "wasted": int(wasted),
                "windows": int(windows),
                "manager_actions": int(total_action_calls),
                "mean_aodt_reward_term": float(np.mean(reward_aodt_terms)) if reward_aodt_terms else 0.0,
                "mean_energy_reward_term": float(np.mean(reward_energy_terms)) if reward_energy_terms else 0.0,
                "uav_window_violation_fraction": float(np.mean(violation_flags)) if violation_flags else 0.0,
                "manager_inference_time_s_total": float(total_select_time),
                "manager_inference_time_s_mean": float(total_select_time / max(total_action_calls, 1)),
                "worker_inference_time_s_total": float(total_worker_select_time),
                "worker_inference_time_s_mean": float(total_worker_select_time / max(total_action_calls, 1)),
                "idle_rate": float(total_worker_idle / max(total_worker_actions, 1)),
                **switching,
                **delay_summary,
            }
        )
        inference_times.append(total_select_time)

    summary = {}
    for key in episode_rows[0].keys():
        values = [row[key] for row in episode_rows]
        first = values[0]
        if isinstance(first, (int, float, np.integer, np.floating, bool)) or first is None or isinstance(first, list):
            summary[key] = safe_mean(values)
    summary["manager_inference_time_s_total"] = float(np.sum(inference_times))
    summary["manager_inference_time_s_mean"] = float(np.mean(inference_times)) if inference_times else 0.0
    summary["worker_inference_time_s_total"] = float(np.mean([row["worker_inference_time_s_total"] for row in episode_rows])) if episode_rows else 0.0
    summary["worker_inference_time_s_mean"] = float(np.mean([row["worker_inference_time_s_mean"] for row in episode_rows])) if episode_rows else 0.0
    summary["idle_rate"] = float(np.mean([row["idle_rate"] for row in episode_rows])) if episode_rows else 0.0
    return summary, episode_rows, transition_rows, transmission_rows


def train_worker_screen(config, output_dir, seeds, episodes, rollout_size, validation_scenarios):
    worker_runs = []
    for seed in seeds:
        completed = load_completed_worker_run(output_dir, seed, config)
        if completed is not None:
            worker_runs.append(completed)
            continue
        seed_config = dict(config)
        seed_config["seed"] = int(seed)
        seed_config["worker_power_mode"] = "fixed_max"
        save_path = output_dir / "models" / f"worker_fixed_max_seed{seed}.pt"
        history_path = output_dir / "history" / f"worker_fixed_max_seed{seed}"
        stem = f"worker_fixed_max_seed{seed}"
        try:
            train_worker(
                config=seed_config,
                num_episodes=episodes,
                rollout_size=rollout_size,
                save_path=save_path,
                fixed_scenario=False,
                validation_interval=max(10, episodes // 10),
                history_output_path=history_path,
            )
            summary, episode_rows, transmission_rows, regret_rows = evaluate_worker_policy(
                policy_name="ppo",
                config=seed_config,
                model_path=save_path,
                scenarios=validation_scenarios,
                compare_same_state=True,
            )
            run = {
                "seed": int(seed),
                "config": seed_config,
                "model_path": str(save_path),
                "history_path": str(history_path),
                "summary": summary,
                "episode_rows": episode_rows,
                "transmission_rows": transmission_rows,
                "regret_rows": regret_rows,
                "checkpoint_metadata": load_checkpoint_metadata(save_path),
                "status": "completed",
                "reused": False,
            }
            worker_runs.append(run)
            write_json(output_dir / "validation" / f"{stem}.json", summary)
            write_csv(output_dir / "validation" / f"{stem}.csv", episode_rows)
            write_csv(output_dir / "validation" / f"{stem}_transmissions.csv", transmission_rows)
            write_csv(output_dir / "validation" / f"{stem}_regret.csv", regret_rows)
            write_run_status(
                output_dir,
                "worker",
                stem,
                {
                    "status": "completed",
                    "seed": int(seed),
                    "model_path": str(save_path),
                    "history_path": str(history_path),
                    "validation_summary_path": str(output_dir / "validation" / f"{stem}.json"),
                },
            )
        except Exception as exc:
            failure = {
                "status": "failed",
                "seed": int(seed),
                "config": seed_config,
                "model_path": str(save_path),
                "history_path": str(history_path),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "run_kind": "worker",
                "identifier": stem,
            }
            write_run_status(output_dir, "worker", stem, failure)
            worker_runs.append(failure)
    return worker_runs


def select_best_worker(worker_runs):
    successful = [run for run in worker_runs if run.get("status", "completed") == "completed" and "summary" in run]
    if not successful:
        raise RuntimeError("No successful worker runs were available for selection.")

    def score(run):
        summary = run["summary"]
        return float(summary["avg_aodt"]) + 0.05 * float(np.mean([row["invalid"] for row in run["episode_rows"]])) + 0.01 * float(np.mean([row["wasted"] for row in run["episode_rows"]]))

    best = min(successful, key=score)
    return best, score(best)


def train_manager_candidate(config, worker_model_path, worker_policy, output_dir, seed, episodes, rollout_size, validation_scenarios):
    seed_config = dict(config)
    seed_config["seed"] = int(seed)
    seed_config["worker_power_mode"] = "fixed_max"
    seed_config["manager_entropy_coef"] = float(config.get("manager_entropy_coef", 0.01))
    seed_config["manager_lr"] = float(config["manager_lr"])
    seed_config["manager_rollout_size"] = int(rollout_size)
    stem = f"manager_{worker_policy}_r{rollout_size}_lr{seed_config['manager_lr']:.0e}_e{seed_config['manager_entropy_coef']:.3g}_seed{seed}"
    save_path = output_dir / "models" / f"{stem}.pt"
    history_path = output_dir / "history" / stem
    completed = load_completed_manager_run(
        output_dir=output_dir,
        worker_policy=worker_policy,
        rollout_size=rollout_size,
        lr=seed_config["manager_lr"],
        entropy_coef=seed_config["manager_entropy_coef"],
        seed=seed,
        config=seed_config,
    )
    if completed is not None:
        return completed
    try:
        train_manager(
            config=seed_config,
            num_episodes=episodes,
            rollout_size=rollout_size,
            worker_model_path=worker_model_path,
            save_path=save_path,
            worker_policy=worker_policy,
            fixed_scenario=False,
            validation_interval=max(10, episodes // 20),
            history_output_path=history_path,
        )
        env = ManagerEnv(
            config=seed_config,
            worker_model_path=worker_model_path,
            worker_policy=worker_policy,
        )
        env.reset(scenario=validation_scenarios[0])
        agent = make_manager_agent(env, seed_config, save_path)
        summary, episode_rows, transition_rows, transmission_rows = evaluate_manager_policy(
            policy_name="ppo",
            env=env,
            scenarios=validation_scenarios,
            config=seed_config,
            agent=agent,
            rng_seed=int(seed) + 11,
        )
        static_summary, static_episode_rows, static_transition_rows, static_transmission_rows = evaluate_manager_policy(
            policy_name="static_heuristic",
            env=env,
            scenarios=validation_scenarios,
            config=seed_config,
            agent=None,
            rng_seed=int(seed) + 11,
        )
        write_json(output_dir / "validation" / f"{stem}.json", summary)
        write_csv(output_dir / "validation" / f"{stem}.csv", episode_rows)
        write_csv(output_dir / "validation" / f"{stem}_transitions.csv", transition_rows)
        write_csv(output_dir / "validation" / f"{stem}_transmissions.csv", transmission_rows)
        write_json(output_dir / "validation" / f"{stem}_static.json", static_summary)
        write_csv(output_dir / "validation" / f"{stem}_static.csv", static_episode_rows)
        write_csv(output_dir / "validation" / f"{stem}_static_transitions.csv", static_transition_rows)
        write_csv(output_dir / "validation" / f"{stem}_static_transmissions.csv", static_transmission_rows)
        run = {
            "seed": int(seed),
            "config": seed_config,
            "model_path": str(save_path),
            "history_path": str(history_path),
            "summary": summary,
            "static_summary": static_summary,
            "episode_rows": episode_rows,
            "transition_rows": transition_rows,
            "transmission_rows": transmission_rows,
            "static_episode_rows": static_episode_rows,
            "static_transition_rows": static_transition_rows,
            "static_transmission_rows": static_transmission_rows,
            "checkpoint_metadata": load_checkpoint_metadata(save_path),
            "status": "completed",
            "reused": False,
        }
        write_run_status(
            output_dir,
            "manager",
            stem,
            {
                "status": "completed",
                "seed": int(seed),
                "model_path": str(save_path),
                "history_path": str(history_path),
                "validation_summary_path": str(output_dir / "validation" / f"{stem}.json"),
                "validation_static_summary_path": str(output_dir / "validation" / f"{stem}_static.json"),
            },
        )
        return run
    except Exception as exc:
        failure = {
            "status": "failed",
            "seed": int(seed),
            "config": seed_config,
            "model_path": str(save_path),
            "history_path": str(history_path),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "run_kind": "manager",
            "identifier": stem,
        }
        write_run_status(output_dir, "manager", stem, failure)
        return failure


def select_manager_candidate(candidate_runs):
    successful = [run for run in candidate_runs if run.get("status", "completed") == "completed" and "summary" in run]
    if not successful:
        raise RuntimeError("No successful manager runs were available for selection.")
    feasible = [run for run in successful if run["summary"]["uav_window_violation_fraction"] <= 1e-9 and run["summary"]["avg_positive_violation"] <= 1e-9]
    if feasible:
        return min(feasible, key=lambda run: run["summary"]["avg_aodt"]), True
    return min(successful, key=lambda run: (run["summary"]["avg_positive_violation"], run["summary"]["avg_aodt"])), False


def aggregate_by_seed(runs):
    keys = [key for key in runs[0].keys() if key not in {"config", "model_path", "history_path", "episode_rows", "window_rows", "static_episode_rows", "static_window_rows", "checkpoint_metadata"}]
    summary = {}
    for key in keys:
        values = [run[key] for run in runs]
        if isinstance(values[0], dict):
            summary[key] = values
        else:
            try:
                summary[key] = float(np.mean(values))
            except Exception:
                summary[key] = values
    return summary


def safe_mean(values):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    if isinstance(filtered[0], list):
        return np.mean(np.asarray(filtered, dtype=np.float32), axis=0).astype(np.float32).tolist()
    return float(np.mean(filtered))


def successful_runs(runs):
    return [run for run in runs if run.get("status", "completed") == "completed" and "summary" in run]


def main():
    parser = argparse.ArgumentParser(description="Phase C2 controller-selection and manager-stability study.")
    parser.add_argument("--output-dir", type=str, default="outputs/results/phase_c2")
    parser.add_argument("--worker-episodes", type=int, default=500)
    parser.add_argument("--manager-episodes", type=int, default=1000)
    parser.add_argument("--worker-rollout-size", type=int, default=512)
    parser.add_argument("--manager-rollout-sizes", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--manager-lrs", nargs="+", type=float, default=[1e-4, 3e-4])
    parser.add_argument("--manager-entropy-coefs", nargs="+", type=float, default=[0.01, 0.005])
    parser.add_argument("--worker-seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--manager-seeds", nargs="+", type=int, default=[51, 52, 53, 54, 55])
    parser.add_argument("--validation-scenarios", type=int, default=40)
    parser.add_argument("--worker-validation-scenarios", type=int, default=40)
    parser.add_argument("--manager-validation-scenarios", type=int, default=40)
    parser.add_argument("--worker-model-path", type=str, default=None)
    parser.add_argument("--worker-only", action="store_true")
    parser.add_argument("--manager-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_phase_c2_config()
    config["validation_worker_scenarios"] = int(args.worker_validation_scenarios)
    config["validation_manager_scenarios"] = int(args.manager_validation_scenarios)
    validation_scenarios = make_scenario_suite(config=config, count=args.validation_scenarios, split="validation")
    write_json(output_dir / "scenario_suite.json", scenario_suite_metadata(validation_scenarios))

    report = {
        "config": config,
        "validation_scenarios": scenario_suite_metadata(validation_scenarios),
        "worker_results": {},
        "manager_screening": {},
        "manager_systems": {},
        "notes": [],
    }
    execution_plan = print_execution_plan(args, config)
    report["execution_plan"] = execution_plan
    write_json(output_dir / "execution_plan.json", execution_plan)
    write_json(output_dir / "plan.json", execution_plan)
    failure_records = []

    selected_worker_run = None
    if not args.manager_only:
        worker_runs = train_worker_screen(
            config=config,
            output_dir=output_dir / "worker",
            seeds=args.worker_seeds,
            episodes=args.worker_episodes,
            rollout_size=args.worker_rollout_size,
            validation_scenarios=validation_scenarios,
        )
        worker_failures = [run for run in worker_runs if run.get("status") == "failed"]
        failure_records.extend(worker_failures)
        successful_worker_runs = successful_runs(worker_runs)
        if not successful_worker_runs:
            raise RuntimeError("All worker runs failed; see status files under the output directory.")
        selected_worker_run, selected_worker_score = select_best_worker(successful_worker_runs)
        greedy_summary, greedy_episode_rows, greedy_transmission_rows, _ = evaluate_worker_policy(
            policy_name="greedy",
            config=config,
            model_path=None,
            scenarios=validation_scenarios,
            compare_same_state=False,
        )

        report["worker_results"] = {
            "ppo_by_seed": [
                {
                    "seed": run["seed"],
                    "model_path": run["model_path"],
                    "summary": run["summary"],
                    "checkpoint_metadata": run["checkpoint_metadata"],
                    "status": run.get("status", "completed"),
                    "reused": bool(run.get("reused", False)),
                }
                for run in worker_runs
            ],
            "selected_worker": {
                "seed": selected_worker_run["seed"],
                "model_path": selected_worker_run["model_path"],
                "summary": selected_worker_run["summary"],
                "score": float(selected_worker_score),
                "checkpoint_metadata": selected_worker_run["checkpoint_metadata"],
            },
            "greedy": {
                "summary": greedy_summary,
                "episode_rows": greedy_episode_rows,
            },
            "failed_runs": worker_failures,
        }

        write_json(output_dir / "worker" / "summary.json", report["worker_results"])
        write_csv(output_dir / "worker" / "greedy_episode_rows.csv", greedy_episode_rows)
        write_csv(output_dir / "worker" / "selected_regret_rows.csv", selected_worker_run["regret_rows"])

    if not args.worker_only:
        if selected_worker_run is None:
            if args.worker_model_path is None:
                selected_worker_run = {
                    "model_path": str(output_dir / "worker" / f"worker_fixed_max_seed{args.worker_seeds[0]}.pt"),
                    "seed": args.worker_seeds[0],
                    "summary": {},
                    "checkpoint_metadata": {},
                }
            else:
                selected_worker_run = {
                    "model_path": str(args.worker_model_path),
                    "seed": int(args.worker_seeds[0]),
                    "summary": {},
                    "checkpoint_metadata": load_checkpoint_metadata(args.worker_model_path),
                }
        worker_model_path = Path(selected_worker_run["model_path"])

        stage_a = []
        for rollout_size in args.manager_rollout_sizes:
            seed_runs = []
            for seed in args.manager_seeds[:3]:
                run = train_manager_candidate(
                    config=config,
                    worker_model_path=worker_model_path,
                    worker_policy="ppo",
                    output_dir=output_dir / "manager_stage_a",
                    seed=seed,
                    episodes=args.manager_episodes,
                    rollout_size=rollout_size,
                    validation_scenarios=validation_scenarios,
                )
                seed_runs.append(run)
                if run.get("status") == "failed":
                    failure_records.append(run)
            successful_seed_runs = successful_runs(seed_runs)
            if not successful_seed_runs:
                raise RuntimeError(f"All manager stage-A runs failed for rollout_size={rollout_size}.")
            stage_a.append(
                {
                    "rollout_size": int(rollout_size),
                    "seed_runs": seed_runs,
                    "summary": aggregate_by_seed([run["summary"] for run in successful_seed_runs]),
                }
            )
        best_rollout = min(stage_a, key=lambda item: item["summary"]["avg_aodt"])["rollout_size"]

        stage_b = []
        for lr in args.manager_lrs:
            seed_runs = []
            for seed in args.manager_seeds[:3]:
                lr_config = dict(config)
                lr_config["manager_lr"] = float(lr)
                run = train_manager_candidate(
                    config=lr_config,
                    worker_model_path=worker_model_path,
                    worker_policy="ppo",
                    output_dir=output_dir / "manager_stage_b",
                    seed=seed,
                    episodes=args.manager_episodes,
                    rollout_size=best_rollout,
                    validation_scenarios=validation_scenarios,
                )
                seed_runs.append(run)
                if run.get("status") == "failed":
                    failure_records.append(run)
            successful_seed_runs = successful_runs(seed_runs)
            if not successful_seed_runs:
                raise RuntimeError(f"All manager stage-B runs failed for lr={lr}.")
            stage_b.append(
                {
                    "lr": float(lr),
                    "seed_runs": seed_runs,
                    "summary": aggregate_by_seed([run["summary"] for run in successful_seed_runs]),
                }
            )
        best_lr = min(stage_b, key=lambda item: item["summary"]["avg_aodt"])["lr"]

        stage_c = []
        for entropy_coef in args.manager_entropy_coefs:
            seed_runs = []
            for seed in args.manager_seeds[:3]:
                entropy_config = dict(config)
                entropy_config["manager_lr"] = float(best_lr)
                entropy_config["manager_entropy_coef"] = float(entropy_coef)
                run = train_manager_candidate(
                    config=entropy_config,
                    worker_model_path=worker_model_path,
                    worker_policy="ppo",
                    output_dir=output_dir / "manager_stage_c",
                    seed=seed,
                    episodes=args.manager_episodes,
                    rollout_size=best_rollout,
                    validation_scenarios=validation_scenarios,
                )
                seed_runs.append(run)
                if run.get("status") == "failed":
                    failure_records.append(run)
            successful_seed_runs = successful_runs(seed_runs)
            if not successful_seed_runs:
                raise RuntimeError(f"All manager stage-C runs failed for entropy_coef={entropy_coef}.")
            stage_c.append(
                {
                    "entropy_coef": float(entropy_coef),
                    "seed_runs": seed_runs,
                    "summary": aggregate_by_seed([run["summary"] for run in successful_seed_runs]),
                }
            )

        stage_c_sorted = sorted(stage_c, key=lambda item: item["summary"]["avg_aodt"])
        finalists = stage_c_sorted[:2] if len(stage_c_sorted) >= 2 else stage_c_sorted
        final_runs = []
        for finalist_index, finalist in enumerate(finalists):
            candidate_config = dict(config)
            candidate_config["manager_lr"] = float(best_lr)
            candidate_config["manager_entropy_coef"] = float(finalist["entropy_coef"])
            candidate_config["manager_rollout_size"] = int(best_rollout)
            seed_runs = []
            for seed in args.manager_seeds:
                run = train_manager_candidate(
                    config=candidate_config,
                    worker_model_path=worker_model_path,
                    worker_policy="ppo",
                    output_dir=output_dir / f"manager_final_candidate_{finalist_index}",
                    seed=seed,
                    episodes=args.manager_episodes,
                    rollout_size=best_rollout,
                    validation_scenarios=validation_scenarios,
                )
                seed_runs.append(run)
                if run.get("status") == "failed":
                    failure_records.append(run)
            successful_seed_runs = successful_runs(seed_runs)
            if not successful_seed_runs:
                raise RuntimeError(f"All final candidate runs failed for candidate_index={finalist_index}.")
            final_runs.append(
                {
                    "candidate_index": finalist_index,
                    "entropy_coef": float(finalist["entropy_coef"]),
                    "seed_runs": seed_runs,
                    "summary": aggregate_by_seed([run["summary"] for run in successful_seed_runs]),
                    "static_summary": aggregate_by_seed([run["static_summary"] for run in successful_seed_runs]),
                }
            )

        greedy_manager_runs = []
        for seed in args.manager_seeds:
            greedy_config = dict(config)
            greedy_config["manager_lr"] = float(best_lr)
            greedy_config["manager_entropy_coef"] = float(stage_c_sorted[0]["entropy_coef"])
            run = train_manager_candidate(
                config=greedy_config,
                worker_model_path=worker_model_path,
                worker_policy="greedy",
                output_dir=output_dir / "manager_greedy_worker",
                seed=seed,
                episodes=args.manager_episodes,
                rollout_size=best_rollout,
                validation_scenarios=validation_scenarios,
            )
            greedy_manager_runs.append(run)
            if run.get("status") == "failed":
                failure_records.append(run)

        report["manager_screening"] = {
            "stage_a": stage_a,
            "stage_b": stage_b,
            "stage_c": stage_c,
            "best_rollout_size": int(best_rollout),
            "best_lr": float(best_lr),
            "finalists": final_runs,
        }
        report["manager_systems"] = {
            "ppo_worker": {
                "selected_worker": selected_worker_run,
                "final_candidates": final_runs,
            },
            "greedy_worker": {
                "runs": [
                    {
                        "seed": run["seed"],
                        "summary": run["summary"],
                        "static_summary": run["static_summary"],
                        "checkpoint_metadata": run["checkpoint_metadata"],
                        "status": run.get("status", "completed"),
                        "reused": bool(run.get("reused", False)),
                    }
                    for run in greedy_manager_runs
                ],
            },
        }
        report["failed_runs"] = failure_records

        write_json(output_dir / "manager" / "summary.json", report["manager_screening"])
        write_json(output_dir / "summary.json", report)

    report_path = Path("docs") / "PHASE_C2_CONTROLLER_SELECTION.md"
    report_path.write_text(
        "# Phase C2 Controller Selection\n\nThis report is generated by `src/dt_uav_v2/evaluation/phase_c2.py`.\n",
        encoding="ascii",
    )


def sample_valid_random_action(env, rng):
    action = []
    power_value = max(env.base_env.sensor_power_levels)
    used = set()
    for m in range(env.num_uavs):
        feasible = []
        for sensor_id in np.where(env.base_env.Q > 0)[0]:
            sensor_id = int(sensor_id)
            if sensor_id in used:
                continue
            packet_size = float(env.base_env.W[sensor_id])
            entity_id = int(env.base_env.sensor_entity[sensor_id])
            dt_host = int(env.base_env.dt_hosts[entity_id])
            uplink_rate = env.base_env.uplink_rate(sensor_id, m, max(env.base_env.sensor_power_levels))
            uplink_delay = packet_size / uplink_rate
            backhaul_delay = 0.0 if m == dt_host else packet_size / env.base_env.backhaul_rate(m, dt_host)
            processing_delay = env.base_env.processing_delay(sensor_id)
            total_delay = uplink_delay + backhaul_delay + processing_delay
            if (
                env.config.get("service_model", "abstract_same_step") == "require_within_slot"
                and total_delay > env.base_env.slot_duration + 1e-6
            ):
                continue
            feasible.append(sensor_id)
        if feasible and rng.random() >= 0.25:
            chosen = int(feasible[rng.integers(low=0, high=len(feasible))])
            action.append((chosen, power_value))
            used.add(chosen)
        else:
            action.append((-1, 0))
    return action


if __name__ == "__main__":
    main()
