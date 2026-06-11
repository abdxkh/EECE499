import argparse
from pathlib import Path

import numpy as np

from dt_uav_v2.agents.manager_agent import ManagerPPOAgent
from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.manager_env import ManagerEnv


def make_manager_agent(env, config, model_path):
    agent = ManagerPPOAgent(
        obs_dim=env.obs_dim,
        num_uavs=env.M,
        num_entities=env.E,
        num_grid_points=env.num_grid_points,
        lr=config["manager_lr"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_eps=config["clip_eps"],
        optimize_backhaul_power=config.get("optimize_backhaul_power", False),
        backhaul_power_min=config.get("backhaul_power_min", config["backhaul_power"]),
        backhaul_power_max=config.get("backhaul_power_max", config["backhaul_power"]),
    )
    agent.load(model_path)

    return agent


def nearest_grid_indices(env, positions):
    distances = np.linalg.norm(
        positions[:, None, :] - env.grid_points[None, :, :],
        axis=2,
    )

    return np.argmin(distances, axis=1).astype(int)


def make_fixed_action(env):
    """
    Keep the reset topology as a fixed deployment baseline.
    """

    return {
        "uav_grid_indices": nearest_grid_indices(env, env.base_env.uav_positions),
        "dt_hosts": env.base_env.dt_hosts.copy().astype(int),
    }


def run_manager_episode(env, policy_name, agent=None, seed=None):
    obs = env.reset(seed=seed)
    done = False

    fixed_action = None
    if policy_name == "fixed":
        fixed_action = make_fixed_action(env)

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

    while not done:
        if policy_name == "ppo":
            action, _, _, _ = agent.select_action(obs, deterministic=True)
        elif policy_name == "random":
            action = env.sample_random_manager_action()
        elif policy_name == "fixed":
            action = fixed_action
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
        windows += 1

    return {
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
    }


def evaluate_manager(
    manager_model_path="outputs/models/manager_backhaul_final.pt",
    worker_model_path="outputs/models/worker_continuous_final.pt",
    episodes=10,
    policies=None,
    worker_policy="ppo",
    seed=None,
    randomize_scenarios=False,
    continuous_worker_power=False,
):
    config = dict(CONFIG)
    if seed is not None:
        config["seed"] = seed
    if continuous_worker_power:
        config["worker_continuous_power"] = True
        config["worker_force_max_power"] = False

    policies = policies or ["ppo", "random", "fixed"]
    manager_model_path = Path(manager_model_path)
    worker_model_path = Path(worker_model_path)

    env = ManagerEnv(
        config=config,
        worker_model_path=worker_model_path,
        worker_policy=worker_policy,
    )
    reset_seed = None if randomize_scenarios else config["seed"]
    obs = env.reset(seed=reset_seed)

    agent = None
    if "ppo" in policies:
        if not manager_model_path.exists():
            raise FileNotFoundError(f"Manager model not found: {manager_model_path}")
        agent = make_manager_agent(env, config, manager_model_path)

    print("Manager evaluation")
    print("Episodes:", episodes)
    print("Policies:", ", ".join(policies))
    print("Manager model:", manager_model_path)
    print("Worker model:", worker_model_path)
    print("Worker policy:", worker_policy)
    print("Observation dim:", len(obs))
    print("Energy budget:", env.energy_budget)
    print("Randomize scenarios:", randomize_scenarios)
    print("Continuous worker power:", config.get("worker_continuous_power", False))
    print("Optimize backhaul power:", config.get("optimize_backhaul_power", False))
    print("-" * 46)
    print(
        f"{'policy':<10}"
        f"{'reward':>12}"
        f"{'aodt':>12}"
        f"{'energy':>12}"
    )
    print("-" * 46)

    all_results = {}

    for policy_name in policies:
        episode_results = []

        for _ in range(episodes):
            episode_seed = None if randomize_scenarios else config["seed"]
            episode_results.append(
                run_manager_episode(
                    env=env,
                    policy_name=policy_name,
                    agent=agent,
                    seed=episode_seed,
                )
            )

        summary = {
            key: float(np.mean([result[key] for result in episode_results]))
            for key in [
                "avg_reward",
                "avg_aodt",
                "tail_aodt",
                "mean_energy",
                "max_energy",
                "violation_rate",
                "avg_queue",
                "final_queue",
                "invalid",
                "wasted",
            ]
        }
        all_results[policy_name] = summary

        print(
            f"{policy_name:<10}"
            f"{summary['avg_reward']:>12.4f}"
            f"{summary['avg_aodt']:>12.4f}"
            f"{summary['mean_energy']:>12.4f}"
        )

    print("-" * 46)

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate manager policies.")
    parser.add_argument("--manager-model-path", type=str, default="outputs/models/manager_backhaul_final.pt")
    parser.add_argument("--worker-model-path", type=str, default="outputs/models/worker_continuous_final.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["ppo", "random", "fixed"],
        choices=["ppo", "random", "fixed"],
    )
    parser.add_argument("--worker-policy", choices=["ppo", "greedy"], default="ppo")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--randomize-scenarios", action="store_true")
    parser.add_argument(
        "--continuous-worker-power",
        action="store_true",
        help="Evaluate using a worker checkpoint trained with continuous power.",
    )

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
        continuous_worker_power=args.continuous_worker_power,
    )


if __name__ == "__main__":
    main()
