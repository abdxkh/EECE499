import argparse
from pathlib import Path

import numpy as np

from dt_uav_v2.agents.ppo import PPOAgent
from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.worker_env import WorkerEnv


def make_worker_agent(env, config, model_path):
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        num_uavs=env.num_uavs,
        num_sensors=env.num_sensors,
        num_power_levels=env.num_power_levels,
        lr=config["worker_lr"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_eps=config["clip_eps"],
        mask_worker_actions=True,
        worker_freshness_bias=config["worker_freshness_bias"],
        force_max_power=config["worker_force_max_power"],
        continuous_power=config.get("worker_continuous_power", False),
        power_min=min(config["sensor_power_levels"]),
        power_max=max(config["sensor_power_levels"]),
    )
    agent.load(model_path)

    return agent


def sample_valid_random_action(env, rng):
    pending = np.where(env.base_env.Q > 0)[0].tolist()
    rng.shuffle(pending)

    action = []
    if env.config.get("worker_continuous_power", False):
        max_power = max(env.base_env.sensor_power_levels)
    else:
        max_power = env.num_power_levels - 1

    for m in range(env.num_uavs):
        if m < len(pending):
            action.append((int(pending[m]), max_power))
        else:
            action.append((-1, 0))

    return action


def run_episode(env, policy_name, rng, agent=None, seed=None):
    obs = env.reset(seed=seed)
    done = False

    rewards = []
    avg_aodt_values = []
    entity_aodt_values = []
    backhaul_energy_values = []
    invalid_count = 0
    wasted_count = 0

    while not done:
        if policy_name == "ppo":
            action, _, _, _ = agent.select_action(obs, deterministic=True)
        elif policy_name == "greedy":
            action = env.sample_aodt_greedy_action()
        elif policy_name == "random":
            action = sample_valid_random_action(env, rng)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

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
    }


def evaluate_worker(
    model_path="outputs/models/worker_continuous_final.pt",
    episodes=10,
    policies=None,
    seed=None,
    randomize_scenarios=False,
    continuous_power=False,
):
    config = dict(CONFIG)
    if seed is not None:
        config["seed"] = seed
    if continuous_power:
        config["worker_continuous_power"] = True
        config["worker_force_max_power"] = False

    policies = policies or ["ppo", "greedy", "random"]
    model_path = Path(model_path)
    rng = np.random.default_rng(config["seed"])

    env = WorkerEnv(config=config)
    reset_seed = None if randomize_scenarios else config["seed"]
    obs = env.reset(seed=reset_seed)

    agent = None
    if "ppo" in policies:
        if not model_path.exists():
            raise FileNotFoundError(f"Worker model not found: {model_path}")
        agent = make_worker_agent(env, config, model_path)

    print("Worker evaluation")
    print("Episodes:", episodes)
    print("Policies:", ", ".join(policies))
    print("Model path:", model_path)
    print("Observation dim:", len(obs))
    print("Randomize scenarios:", randomize_scenarios)
    print("Continuous power:", config.get("worker_continuous_power", False))
    print("-" * 46)
    print(
        f"{'policy':<10}"
        f"{'avg_reward':>12}"
        f"{'avg_aodt':>12}"
        f"{'bh_energy':>12}"
    )
    print("-" * 46)

    all_results = {}

    for policy_name in policies:
        episode_results = []

        for episode in range(episodes):
            if randomize_scenarios:
                episode_seed = None
            else:
                episode_seed = config["seed"]

            result = run_episode(
                env=env,
                policy_name=policy_name,
                rng=rng,
                agent=agent,
                seed=episode_seed,
            )
            episode_results.append(result)

        summary = {
            key: float(np.mean([result[key] for result in episode_results]))
            for key in [
                "avg_reward",
                "avg_aodt",
                "tail_aodt_95",
                "avg_backhaul_energy",
                "invalid",
                "wasted",
            ]
        }
        all_results[policy_name] = summary

        print(
            f"{policy_name:<10}"
            f"{summary['avg_reward']:>12.4f}"
            f"{summary['avg_aodt']:>12.4f}"
            f"{summary['avg_backhaul_energy']:>12.4f}"
        )

    print("-" * 46)

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate worker policies.")
    parser.add_argument("--model-path", type=str, default="outputs/models/worker_continuous_final.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["ppo", "greedy", "random"],
        choices=["ppo", "greedy", "random"],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--randomize-scenarios", action="store_true")
    parser.add_argument(
        "--continuous-power",
        action="store_true",
        help="Evaluate a worker checkpoint trained with continuous power.",
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
        continuous_power=args.continuous_power,
    )


if __name__ == "__main__":
    main()
