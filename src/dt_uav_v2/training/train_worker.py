import argparse
from pathlib import Path

import numpy as np
import torch

from dt_uav_v2.agents.ppo import PPOAgent, RolloutMemory
from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.worker_env import WorkerEnv


def train_worker(
    config=None,
    num_episodes=50,
    rollout_size=512,
    save_path="outputs/models/worker_continuous_final.pt",
    fixed_scenario=True,
):
    """
    Train the slot-level worker PPO policy.

    This function only trains the worker. It does not touch manager placement,
    Lyapunov queues, evaluation baselines, or plotting.
    """

    config = dict(CONFIG if config is None else config)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    env = WorkerEnv(config=config)
    reset_seed = config["seed"] if fixed_scenario else None
    obs = env.reset(seed=reset_seed)

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

    memory = RolloutMemory()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_avg_aodt = float("inf")

    print("Starting worker PPO training")
    print("Episodes:", num_episodes)
    print("Rollout size:", rollout_size)
    print("Observation dim:", env.obs_dim)
    print("UAVs:", env.num_uavs)
    print("Sensors:", env.num_sensors)
    print("Power levels:", env.num_power_levels)
    print("Fixed scenario:", fixed_scenario)
    print("Worker freshness bias:", config["worker_freshness_bias"])
    print("Continuous power:", config.get("worker_continuous_power", False))
    print("Force max power:", config["worker_force_max_power"])
    print("-" * 60)

    for episode in range(1, num_episodes + 1):
        obs = env.reset(seed=reset_seed)
        done = False

        episode_rewards = []
        episode_aodt = []
        episode_invalid = 0
        episode_wasted = 0
        episode_backhaul_energy = []
        last_update_stats = {}

        while not done:
            action, action_indices, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)

            memory.add(
                obs=obs,
                action_indices=action_indices,
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=value,
            )

            episode_rewards.append(reward)
            episode_aodt.append(info["avg_aodt"])
            episode_invalid += info["invalid_count"]
            episode_wasted += info["wasted_count"]
            episode_backhaul_energy.append(float(np.mean(info["backhaul_energy"])))

            obs = next_obs

            if len(memory) >= rollout_size or done:
                last_value = 0.0

                if not done:
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(
                            obs,
                            dtype=torch.float32,
                            device=agent.device,
                        ).view(1, -1)
                        _, _, value_tensor = agent.model(obs_tensor)
                        last_value = float(value_tensor.item())

                last_update_stats = agent.update(memory, last_value=last_value)
                memory.clear()

        avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        avg_aodt = float(np.mean(episode_aodt)) if episode_aodt else 0.0
        avg_backhaul_energy = (
            float(np.mean(episode_backhaul_energy))
            if episode_backhaul_energy
            else 0.0
        )

        loss_text = ""
        if last_update_stats:
            loss_text = (
                f" | loss={last_update_stats['loss']:.4f}"
                f" actor={last_update_stats['actor_loss']:.4f}"
                f" critic={last_update_stats['critic_loss']:.4f}"
            )

        if avg_aodt < best_avg_aodt:
            best_avg_aodt = avg_aodt
            agent.save(save_path)
            best_text = " | saved_best=True"
        else:
            best_text = " | saved_best=False"

        print(
            f"Episode {episode:04d} | "
            f"avg_reward={avg_reward:.4f} | "
            f"avg_aodt={avg_aodt:.4f} | "
            f"invalid={episode_invalid} | "
            f"wasted={episode_wasted} | "
            f"avg_backhaul_energy={avg_backhaul_energy:.4f}"
            f"{loss_text}"
            f"{best_text}"
        )

    print("-" * 60)
    print("Saved best worker model:", save_path)
    print("Best average AoDT:", f"{best_avg_aodt:.4f}")

    return agent


def parse_args():
    parser = argparse.ArgumentParser(description="Train worker PPO policy.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--rollout-size", type=int, default=512)
    parser.add_argument("--save-path", type=str, default="outputs/models/worker_continuous_final.pt")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--randomize-scenarios",
        action="store_true",
        help="Randomize positions, placement, and packet sequence across episodes.",
    )
    parser.add_argument(
        "--continuous-power",
        action="store_true",
        help="Use a continuous worker power policy in watts instead of discrete power indices.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = dict(CONFIG)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.continuous_power:
        config["worker_continuous_power"] = True
        config["worker_force_max_power"] = False

    train_worker(
        config=config,
        num_episodes=args.episodes,
        rollout_size=args.rollout_size,
        save_path=args.save_path,
        fixed_scenario=not args.randomize_scenarios,
    )


if __name__ == "__main__":
    main()
