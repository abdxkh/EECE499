import argparse
from pathlib import Path

import numpy as np
import torch

from dt_uav_v2.agents.manager_agent import ManagerPPOAgent, ManagerRolloutMemory
from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.manager_env import ManagerEnv


def train_manager(
    config=None,
    num_episodes=50,
    rollout_size=64,
    worker_model_path="outputs/models/worker_continuous_final.pt",
    save_path="outputs/models/manager_backhaul_final.pt",
    worker_policy="ppo",
    fixed_scenario=True,
):
    """
    Train the slow-timescale manager PPO policy on top of a frozen worker.
    """

    config = dict(CONFIG if config is None else config)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    env = ManagerEnv(
        config=config,
        worker_model_path=worker_model_path,
        worker_policy=worker_policy,
    )
    reset_seed = config["seed"] if fixed_scenario else None
    obs = env.reset(seed=reset_seed)

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

    memory = ManagerRolloutMemory()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_score = float("inf")

    print("Starting manager PPO training")
    print("Episodes:", num_episodes)
    print("Rollout size:", rollout_size)
    print("Observation dim:", env.obs_dim)
    print("UAVs:", env.M)
    print("Entities:", env.E)
    print("Grid points:", env.num_grid_points)
    print("Manager horizon:", env.H)
    print("Worker policy:", worker_policy)
    print("Worker model:", worker_model_path)
    print("Fixed scenario:", fixed_scenario)
    print("Optimize backhaul power:", config.get("optimize_backhaul_power", False))
    print("-" * 80)

    for episode in range(1, num_episodes + 1):
        obs = env.reset(seed=reset_seed)
        done = False

        episode_rewards = []
        episode_aodt = []
        episode_tail = []
        episode_energy = []
        episode_queue = []
        episode_invalid = 0
        episode_wasted = 0
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
            episode_aodt.append(info["avg_window_aodt"])
            episode_tail.append(info["tail_window_aodt"])
            episode_energy.append(float(np.mean(info["avg_energy_per_uav"])))
            episode_queue.append(float(np.mean(info["virtual_queues"])))
            episode_invalid += info["invalid_count"]
            episode_wasted += info["wasted_count"]

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
                        _, _, _, value_tensor = agent.model(obs_tensor)
                        last_value = float(value_tensor.item())

                last_update_stats = agent.update(memory, last_value=last_value)
                memory.clear()

        avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        avg_aodt = float(np.mean(episode_aodt)) if episode_aodt else 0.0
        avg_tail = float(np.mean(episode_tail)) if episode_tail else 0.0
        avg_energy = float(np.mean(episode_energy)) if episode_energy else 0.0
        avg_queue = float(np.mean(episode_queue)) if episode_queue else 0.0

        # Score used only for checkpoint selection: lower AoDT and lower queue.
        checkpoint_score = avg_aodt + avg_queue
        if checkpoint_score < best_score:
            best_score = checkpoint_score
            agent.save(save_path)
            best_text = " | saved_best=True"
        else:
            best_text = " | saved_best=False"

        loss_text = ""
        if last_update_stats:
            loss_text = (
                f" | loss={last_update_stats['loss']:.4f}"
                f" actor={last_update_stats['actor_loss']:.4f}"
                f" critic={last_update_stats['critic_loss']:.4f}"
            )

        print(
            f"Episode {episode:04d} | "
            f"avg_reward={avg_reward:.4f} | "
            f"avg_aodt={avg_aodt:.4f} | "
            f"tail={avg_tail:.4f} | "
            f"energy={avg_energy:.4f} | "
            f"queue={avg_queue:.4f} | "
            f"invalid={episode_invalid} | "
            f"wasted={episode_wasted}"
            f"{loss_text}"
            f"{best_text}"
        )

    print("-" * 80)
    print("Saved best manager model:", save_path)
    print("Best checkpoint score:", f"{best_score:.4f}")

    return agent


def parse_args():
    parser = argparse.ArgumentParser(description="Train manager PPO policy.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--rollout-size", type=int, default=64)
    parser.add_argument("--worker-model-path", type=str, default="outputs/models/worker_continuous_final.pt")
    parser.add_argument("--save-path", type=str, default="outputs/models/manager_backhaul_final.pt")
    parser.add_argument("--worker-policy", choices=["ppo", "greedy"], default="ppo")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--randomize-scenarios", action="store_true")
    parser.add_argument(
        "--continuous-worker-power",
        action="store_true",
        help="Load a worker checkpoint trained with continuous power.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = dict(CONFIG)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.continuous_worker_power:
        config["worker_continuous_power"] = True
        config["worker_force_max_power"] = False

    train_manager(
        config=config,
        num_episodes=args.episodes,
        rollout_size=args.rollout_size,
        worker_model_path=args.worker_model_path,
        save_path=args.save_path,
        worker_policy=args.worker_policy,
        fixed_scenario=not args.randomize_scenarios,
    )


if __name__ == "__main__":
    main()
