import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dt_uav_v2.agents.manager_agent import ManagerPPOAgent, ManagerRolloutMemory
from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.manager_env import ManagerEnv
from dt_uav_v2.utils.scenarios import make_scenario_suite, scenario_suite_metadata


def run_manager_validation(agent, config, scenarios, worker_model_path, worker_policy):
    env = ManagerEnv(
        config=config,
        worker_model_path=worker_model_path,
        worker_policy=worker_policy,
    )
    metrics = []
    uav_window_flags = []
    positive_violations = []

    for scenario in scenarios:
        obs = env.reset(scenario=scenario)
        done = False
        rewards = []
        aodt_values = []
        energy_values = []
        positive_violations = []
        queue_values = []

        while not done:
            action, _, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            aodt_values.append(info["avg_window_aodt"])
            energy = np.asarray(info["avg_energy_per_uav"], dtype=np.float32)
            energy_values.extend(energy.tolist())
            positive_violations.extend(info["positive_violation"].tolist())
            uav_window_flags.extend((energy > float(config["backhaul_energy_budget"])).astype(np.float32).tolist())
            queue_values.extend(info["virtual_queues"].tolist())

        metrics.append(
            {
                "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
                "avg_aodt": float(np.mean(aodt_values)) if aodt_values else 0.0,
                "avg_energy": float(np.mean(energy_values)) if energy_values else 0.0,
                "avg_positive_violation": float(np.mean(positive_violations)) if positive_violations else 0.0,
                "avg_queue": float(np.mean(queue_values)) if queue_values else 0.0,
            }
        )

    summary = {
        key: float(np.mean([metric[key] for metric in metrics]))
        for key in [
            "avg_reward",
            "avg_aodt",
            "avg_energy",
            "avg_positive_violation",
            "avg_queue",
        ]
    }
    summary["avg_positive_violation"] = float(np.mean(positive_violations)) if positive_violations else 0.0
    summary["uav_window_violation_fraction"] = float(np.mean(uav_window_flags)) if uav_window_flags else 0.0
    summary["feasible"] = bool(
        summary["avg_positive_violation"] <= 1e-9
        and summary["uav_window_violation_fraction"] <= 1e-9
    )
    if summary["feasible"]:
        summary["selection_score"] = summary["avg_aodt"]
    else:
        summary["selection_score"] = 1000.0 * summary["avg_positive_violation"] + summary["avg_aodt"]
    summary["score"] = summary["avg_aodt"] + 10.0 * summary["avg_positive_violation"] + 0.1 * summary["avg_queue"]
    return summary


def train_manager(
    config=None,
    num_episodes=50,
    rollout_size=64,
    worker_model_path="outputs/models/worker_continuous_final.pt",
    save_path="outputs/models/manager_backhaul_final.pt",
    worker_policy="ppo",
    fixed_scenario=True,
    validation_interval=10,
    history_output_path=None,
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
    training_scenario = None
    if fixed_scenario:
        training_scenario = make_scenario_suite(config=config, count=1, split="train")[0]
    obs = env.reset(seed=reset_seed, scenario=training_scenario)

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
        entropy_coef=config.get("manager_entropy_coef", 0.01),
        optimize_backhaul_power=config.get("optimize_backhaul_power", False),
        backhaul_power_min=config.get("backhaul_power_min", config["backhaul_power"]),
        backhaul_power_max=config.get("backhaul_power_max", config["backhaul_power"]),
        host_action_mode=config.get("manager_host_action_mode", "feasible_enum"),
    )

    validation_scenarios = make_scenario_suite(
        config=config,
        count=int(config.get("validation_manager_scenarios", 8)),
        split="validation",
    )

    memory = ManagerRolloutMemory()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    history_rows = []
    validation_rows = []

    print("Starting manager PPO training")
    print("Episodes:", num_episodes)
    print("Rollout size:", rollout_size)
    print("Observation dim:", env.obs_dim)
    print("UAVs:", env.M)
    print("Entities:", env.E)
    print("Grid points:", env.num_grid_points)
    print("Total DT assignments:", env.total_dt_assignments)
    print("Manager horizon:", env.H)
    print("Worker policy:", worker_policy)
    print("Worker model:", worker_model_path)
    print("Reward mode:", config.get("manager_reward_mode", "queue_weighted_energy"))
    print("Host action mode:", config.get("manager_host_action_mode", "feasible_enum"))
    print("Validation scenarios:", len(validation_scenarios))
    print("-" * 96)

    for episode in range(1, num_episodes + 1):
        obs = env.reset(seed=reset_seed, scenario=training_scenario if fixed_scenario else None)
        done = False

        episode_rewards = []
        episode_aodt = []
        episode_tail = []
        episode_energy = []
        episode_queue = []
        episode_invalid = 0
        episode_wasted = 0
        total_switches = 0
        total_movement = 0.0
        total_dt_switches = 0
        chosen_backhaul_powers = []
        reward_aodt_terms = []
        reward_energy_terms = []
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
            total_switches += info["uav_switches"]
            total_movement += info["movement_distance"]
            total_dt_switches += info["dt_switches"]
            chosen_backhaul_powers.extend(info["backhaul_powers"].tolist())
            reward_aodt_terms.append(float(info["reward_terms"]["aodt_term"]))
            reward_energy_terms.append(float(info["reward_terms"]["energy_term"]))

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
        mean_backhaul_power = float(np.mean(chosen_backhaul_powers)) if chosen_backhaul_powers else 0.0

        validation = None
        saved_best = False
        if episode % max(int(validation_interval), 1) == 0 or episode == num_episodes:
            validation = run_manager_validation(
                agent=agent,
                config=config,
                scenarios=validation_scenarios,
                worker_model_path=worker_model_path,
                worker_policy=worker_policy,
            )
            if validation["selection_score"] < best_score:
                best_score = validation["selection_score"]
                agent.save_checkpoint(
                    save_path,
                    extra_metadata={
                        "config": config,
                        "seed": int(config["seed"]),
                        "obs_spec_version": config.get("obs_spec_version"),
                        "env_version": config.get("env_version"),
                        "scenario_distribution_version": config.get("scenario_distribution_version"),
                        "slot_duration": float(config.get("slot_duration")),
                        "manager_horizon": int(config.get("manager_horizon")),
                        "action_mode": config.get("manager_host_action_mode", "feasible_enum"),
                        "reward_mode": config.get("manager_reward_mode", "queue_weighted_energy"),
                        "worker_power_mode": config.get("worker_power_mode", "learned_beta"),
                        "worker_context_mode": config.get("worker_context_mode", "random_feasible_context"),
                        "architecture_variant": "manager_feasible_enum",
                        "training_step": int(episode),
                        "validation_metrics": validation,
                        "scenario_suite": scenario_suite_metadata(validation_scenarios),
                        "worker_model_path": str(worker_model_path),
                        "validation_selection_score": float(validation["selection_score"]),
                    },
                )
                saved_best = True
        best_text = f" | saved_best={str(saved_best)}"

        loss_text = ""
        if last_update_stats:
            loss_text = (
                f" | loss={last_update_stats['loss']:.4f}"
                f" actor={last_update_stats['actor_loss']:.4f}"
                f" critic={last_update_stats['critic_loss']:.4f}"
                f" entropy={last_update_stats['entropy']:.4f}"
                f" kl={last_update_stats['approx_kl']:.4f}"
                f" clip={last_update_stats['clip_fraction']:.4f}"
                f" ev={last_update_stats['explained_variance']:.4f}"
                f" grad={last_update_stats['grad_norm']:.4f}"
            )

        history_rows.append(
            {
                "episode": int(episode),
                "train_reward": avg_reward,
                "train_aodt": avg_aodt,
                "train_tail_aodt": avg_tail,
                "train_energy": avg_energy,
                "train_queue": avg_queue,
                "uav_switches": int(total_switches),
                "movement_distance": total_movement,
                "dt_switches": int(total_dt_switches),
                "backhaul_power_mean": mean_backhaul_power,
                "reward_aodt_term_mean": float(np.mean(reward_aodt_terms)) if reward_aodt_terms else 0.0,
                "reward_energy_term_mean": float(np.mean(reward_energy_terms)) if reward_energy_terms else 0.0,
                "update_loss": None if not last_update_stats else last_update_stats["loss"],
                "update_actor_loss": None if not last_update_stats else last_update_stats["actor_loss"],
                "update_critic_loss": None if not last_update_stats else last_update_stats["critic_loss"],
                "update_entropy": None if not last_update_stats else last_update_stats["entropy"],
                "update_approx_kl": None if not last_update_stats else last_update_stats["approx_kl"],
                "update_clip_fraction": None if not last_update_stats else last_update_stats["clip_fraction"],
                "update_explained_variance": None if not last_update_stats else last_update_stats["explained_variance"],
                "update_grad_norm": None if not last_update_stats else last_update_stats["grad_norm"],
                "val_reward": None if validation is None else validation["avg_reward"],
                "val_aodt": None if validation is None else validation["avg_aodt"],
                "val_energy": None if validation is None else validation["avg_energy"],
                "val_pos_violation": None if validation is None else validation["avg_positive_violation"],
                "val_queue": None if validation is None else validation["avg_queue"],
                "val_uav_window_violation_fraction": None if validation is None else validation["uav_window_violation_fraction"],
                "val_feasible": None if validation is None else validation["feasible"],
                "val_selection_score": None if validation is None else validation["selection_score"],
                "val_score": None if validation is None else validation["score"],
                "saved_best": saved_best,
            }
        )

        val_text = " | val_skipped=True"
        if validation is not None:
            val_text = (
                f" | val_aodt={validation['avg_aodt']:.4f} | "
                f"val_pos_viol={validation['avg_positive_violation']:.4f}"
            )
            validation_rows.append(
                {
                    "episode": int(episode),
                    "validation_score": float(validation["score"]),
                    "avg_reward": float(validation["avg_reward"]),
                    "avg_aodt": float(validation["avg_aodt"]),
                    "avg_energy": float(validation["avg_energy"]),
                    "avg_positive_violation": float(validation["avg_positive_violation"]),
                    "avg_queue": float(validation["avg_queue"]),
                }
            )

        print(
            f"Episode {episode:04d} | "
            f"train_reward={avg_reward:.4f} | "
            f"train_aodt={avg_aodt:.4f} | "
            f"tail={avg_tail:.4f} | "
            f"energy={avg_energy:.4f} | "
            f"queue={avg_queue:.4f} | "
            f"uav_switches={total_switches} | "
            f"movement={total_movement:.4f} | "
            f"dt_switches={total_dt_switches} | "
            f"bh_power={mean_backhaul_power:.4f} | "
            f"{val_text}"
            f"{loss_text}"
            f"{best_text}"
        )

    print("-" * 96)
    print("Saved best manager model:", save_path)
    print("Best validation score:", f"{best_score:.4f}")

    if history_output_path is not None:
        history_output_path = Path(history_output_path)
        history_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_output_path.with_suffix(".json"), "w", encoding="ascii") as fh:
            json.dump(history_rows, fh, indent=2)
        with open(history_output_path.with_suffix(".csv"), "w", newline="", encoding="ascii") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)
        validation_history_path = history_output_path.with_name(history_output_path.name + "_validation")
        with open(validation_history_path.with_suffix(".json"), "w", encoding="ascii") as fh:
            json.dump(validation_rows, fh, indent=2)
        if validation_rows:
            with open(validation_history_path.with_suffix(".csv"), "w", newline="", encoding="ascii") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(validation_rows[0].keys()))
                writer.writeheader()
                writer.writerows(validation_rows)

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
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--history-output-path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = dict(CONFIG)

    if args.seed is not None:
        config["seed"] = args.seed

    train_manager(
        config=config,
        num_episodes=args.episodes,
        rollout_size=args.rollout_size,
        worker_model_path=args.worker_model_path,
        save_path=args.save_path,
        worker_policy=args.worker_policy,
        fixed_scenario=not args.randomize_scenarios,
        validation_interval=args.validation_interval,
        history_output_path=args.history_output_path,
    )


if __name__ == "__main__":
    main()
