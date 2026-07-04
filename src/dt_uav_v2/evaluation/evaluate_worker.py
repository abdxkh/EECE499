import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dt_uav_v2.agents.ppo import PPOAgent
from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.worker_env import WorkerEnv
from dt_uav_v2.training.train_worker import apply_worker_context, prepare_worker_config
from dt_uav_v2.utils.scenarios import make_scenario_suite, sample_worker_context, scenario_suite_metadata


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
        power_mode=config.get("worker_power_mode", "learned_beta"),
        force_max_power=config["worker_force_max_power"],
        continuous_power=config.get("worker_continuous_power", False),
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
        service_model=config.get("service_model", "abstract_same_step"),
    )
    agent.load(model_path)
    return agent


def sample_valid_random_action(env, rng):
    action = []
    power_value = max(env.base_env.sensor_power_levels) if env.config.get("worker_continuous_power", False) else env.num_power_levels - 1
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


def run_episode(env, policy_name, rng, agent=None):
    done = False
    rewards = []
    avg_aodt_values = []
    entity_aodt_values = []
    backhaul_energy_values = []
    invalid_count = 0
    wasted_count = 0
    selected_powers = []

    obs = env._state_to_obs(env.base_env.get_basic_state())

    while not done:
        if policy_name == "ppo":
            action, _, _, _ = agent.select_action(obs, deterministic=True)
        elif policy_name == "greedy":
            action = env.sample_max_age_reduction_action()
        elif policy_name == "proximity":
            action = env.sample_proximity_greedy_action()
        elif policy_name == "random":
            action = sample_valid_random_action(env, rng)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        for sensor_id, power in action:
            if sensor_id != -1:
                selected_powers.append(float(power))

        obs, reward, done, info = env.step(action)

        rewards.append(reward)
        avg_aodt_values.append(info["avg_aodt"])
        entity_aodt_values.extend(info["entity_aodt"].tolist())
        backhaul_energy_values.append(float(np.mean(info["backhaul_energy"])))
        invalid_count += info["invalid_count"]
        wasted_count += info["wasted_count"]

    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_aodt": float(np.mean(avg_aodt_values)),
        "tail_aodt_95": float(np.percentile(entity_aodt_values, 95)),
        "avg_backhaul_energy": float(np.mean(backhaul_energy_values)),
        "invalid": int(invalid_count),
        "wasted": int(wasted_count),
        "mean_uplink_power": float(np.mean(selected_powers)) if selected_powers else 0.0,
    }


def write_results(output_dir, scenario_metadata, per_episode_rows, summary):
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


def evaluate_worker(
    model_path="outputs/models/worker_continuous_final.pt",
    episodes=10,
    policies=None,
    seed=None,
    randomize_scenarios=False,
    output_dir=None,
    power_mode=None,
    worker_reward_mode=None,
):
    config = prepare_worker_config(dict(CONFIG))
    if seed is not None:
        config["seed"] = seed
    if power_mode is not None:
        config["worker_power_mode"] = power_mode
    if worker_reward_mode is not None:
        config["worker_reward_mode"] = worker_reward_mode
    config = prepare_worker_config(config)

    policies = policies or ["ppo", "greedy", "proximity", "random"]
    model_path = Path(model_path)

    env = WorkerEnv(config=config)
    initial_scenario = make_scenario_suite(config=config, count=1, split="test")[0]
    env.reset(scenario=initial_scenario)
    initial_context = sample_worker_context(
        env.base_env,
        mode=config.get("worker_context_mode", "random_feasible_context"),
        rng=np.random.default_rng(int(initial_scenario["scenario_seed"]) + 123),
        config=config,
    )
    obs = apply_worker_context(env, initial_context)

    agent = None
    if "ppo" in policies:
        if not model_path.exists():
            raise FileNotFoundError(f"Worker model not found: {model_path}")
        agent = make_worker_agent(env, config, model_path)

    if randomize_scenarios:
        scenarios = make_scenario_suite(config=config, count=episodes, split="test")
    else:
        scenarios = [make_scenario_suite(config=config, count=1, split="test")[0] for _ in range(episodes)]

    print("Worker evaluation")
    print("Episodes:", episodes)
    print("Policies:", ", ".join(policies))
    print("Model path:", model_path)
    print("Observation dim:", len(obs))
    print("Randomize scenarios:", randomize_scenarios)
    print("Power mode:", config.get("worker_power_mode", "learned_beta"))
    print("Worker reward mode:", config.get("worker_reward_mode", "current"))
    print("-" * 52)
    print(f"{'policy':<10}{'avg_reward':>12}{'avg_aodt':>12}{'bh_energy':>12}")
    print("-" * 52)

    all_results = {}
    per_episode_rows = []
    policy_seed_offsets = {
        "ppo": 0,
        "greedy": 1_000,
        "proximity": 2_000,
        "random": 3_000,
    }

    for policy_name in policies:
        episode_results = []
        policy_rng = np.random.default_rng(config["seed"] + policy_seed_offsets[policy_name])

        for episode_index, scenario in enumerate(scenarios):
            env.reset(scenario=scenario)
            context = sample_worker_context(
                env.base_env,
                mode=config.get("worker_context_mode", "random_feasible_context"),
                rng=np.random.default_rng(int(scenario["scenario_seed"]) + 123),
                config=config,
            )
            apply_worker_context(env, context)
            result = run_episode(
                env=env,
                policy_name=policy_name,
                rng=policy_rng,
                agent=agent,
            )
            episode_results.append(result)
            per_episode_rows.append(
                {
                    "policy": policy_name,
                    "episode_index": episode_index,
                    "scenario_seed": int(scenario["scenario_seed"]),
                    **result,
                }
            )

        summary = {
            key: float(np.mean([result[key] for result in episode_results]))
            for key in episode_results[0].keys()
        }
        all_results[policy_name] = summary

        print(
            f"{policy_name:<10}"
            f"{summary['avg_reward']:>12.4f}"
            f"{summary['avg_aodt']:>12.4f}"
            f"{summary['avg_backhaul_energy']:>12.4f}"
        )

    print("-" * 52)

    if output_dir is not None:
        write_results(
            output_dir=Path(output_dir),
            scenario_metadata=scenario_suite_metadata(scenarios),
            per_episode_rows=per_episode_rows,
            summary=all_results,
        )

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate worker policies.")
    parser.add_argument("--model-path", type=str, default="outputs/models/worker_continuous_final.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["ppo", "greedy", "proximity", "random"],
        choices=["ppo", "greedy", "proximity", "random"],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--randomize-scenarios", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--power-mode", choices=["learned_beta", "fixed_max", "fixed_mid"], default=None)
    parser.add_argument(
        "--worker-reward-mode",
        choices=["current"],
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_worker(
        model_path=args.model_path,
        episodes=args.episodes,
        policies=args.policies,
        seed=args.seed,
        randomize_scenarios=args.randomize_scenarios,
        output_dir=args.output_dir,
        power_mode=args.power_mode,
        worker_reward_mode=args.worker_reward_mode,
    )


if __name__ == "__main__":
    main()
