import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dt_uav_v2.agents.ppo import PPOAgent, RolloutMemory
from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.worker_env import WorkerEnv
from dt_uav_v2.utils.scenarios import make_scenario_suite, sample_worker_context, scenario_suite_metadata


def prepare_worker_config(config):
    config = dict(CONFIG if config is None else config)
    power_mode = config.get("worker_power_mode", "learned_beta")
    if power_mode not in {"learned_beta", "fixed_max", "fixed_mid"}:
        raise ValueError(f"Unknown worker power mode: {power_mode}")
    reward_mode = config.get("worker_reward_mode", "current")
    if reward_mode != "current":
        raise ValueError(f"Unknown worker reward mode: {reward_mode}")
    config["worker_continuous_power"] = True
    config["worker_force_max_power"] = False
    return config


def apply_worker_context(env, context):
    env.base_env.apply_manager_context(
        uav_positions=context["uav_positions"],
        dt_hosts=context["dt_hosts"],
        backhaul_powers=context["backhaul_powers"],
    )
    obs = env._state_to_obs(env.base_env.get_basic_state())
    env.obs_dim = len(obs)
    return obs


def run_worker_validation(agent, config, scenarios):
    env = WorkerEnv(config=config)
    results = []

    for scenario in scenarios:
        env.reset(scenario=scenario)
        context_rng = np.random.default_rng(int(scenario["scenario_seed"]) + 123)
        context = sample_worker_context(
            env.base_env,
            mode=config.get("worker_context_mode", "random_feasible_context"),
            rng=context_rng,
            config=config,
        )
        obs = apply_worker_context(env, context)
        done = False
        rewards = []
        aodt_values = []
        invalid = 0
        wasted = 0

        while not done:
            action, _, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            aodt_values.append(info["avg_aodt"])
            invalid += info["invalid_count"]
            wasted += info["wasted_count"]

        results.append(
            {
                "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
                "avg_aodt": float(np.mean(aodt_values)) if aodt_values else 0.0,
                "invalid": float(invalid),
                "wasted": float(wasted),
            }
        )

    summary = {
        key: float(np.mean([result[key] for result in results]))
        for key in ["avg_reward", "avg_aodt", "invalid", "wasted"]
    }
    summary["score"] = summary["avg_aodt"] + 0.05 * summary["invalid"] + 0.01 * summary["wasted"]
    return summary


def train_worker(
    config=None,
    num_episodes=50,
    rollout_size=512,
    save_path="outputs/models/worker_continuous_final.pt",
    fixed_scenario=True,
    validation_interval=10,
    history_output_path=None,
):
    """
    Train the slot-level worker PPO policy with reproducible validation.
    """

    config = prepare_worker_config(config)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    env = WorkerEnv(config=config)
    reset_seed = config["seed"] if fixed_scenario else None
    training_scenario = None
    if fixed_scenario:
        training_scenario = make_scenario_suite(config=config, count=1, split="train")[0]

    obs = env.reset(seed=reset_seed, scenario=training_scenario)
    initial_context = sample_worker_context(
        env.base_env,
        mode=config.get("worker_context_mode", "random_feasible_context"),
        rng=np.random.default_rng(config["seed"]),
        config=config,
    )
    obs = apply_worker_context(env, initial_context)

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

    validation_scenarios = make_scenario_suite(
        config=config,
        count=int(config.get("validation_worker_scenarios", 8)),
        split="validation",
    )

    memory = RolloutMemory()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_score = float("inf")
    history_rows = []
    validation_rows = []

    print("Starting worker PPO training")
    print("Episodes:", num_episodes)
    print("Rollout size:", rollout_size)
    print("Observation dim:", env.obs_dim)
    print("UAVs:", env.num_uavs)
    print("Sensors:", env.num_sensors)
    print("Power mode:", config.get("worker_power_mode", "learned_beta"))
    print("Worker reward mode:", config.get("worker_reward_mode", "current"))
    print("Worker context mode:", config.get("worker_context_mode", "random_feasible_context"))
    print("Fixed scenario:", fixed_scenario)
    print("Validation scenarios:", len(validation_scenarios))
    print("-" * 72)

    for episode in range(1, num_episodes + 1):
        scenario = training_scenario if fixed_scenario else None
        obs = env.reset(seed=reset_seed, scenario=scenario)
        if fixed_scenario:
            context_rng = np.random.default_rng(config["seed"])
        else:
            context_rng = np.random.default_rng()
        context = sample_worker_context(
            env.base_env,
            mode=config.get("worker_context_mode", "random_feasible_context"),
            rng=context_rng,
            config=config,
        )
        obs = apply_worker_context(env, context)

        done = False
        episode_rewards = []
        episode_aodt = []
        episode_invalid = 0
        episode_wasted = 0
        episode_backhaul_energy = []
        selected_powers = []
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

            for sensor_id, power in action:
                if sensor_id != -1:
                    selected_powers.append(float(power))

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
        avg_backhaul_energy = float(np.mean(episode_backhaul_energy)) if episode_backhaul_energy else 0.0
        selected_powers = np.asarray(selected_powers, dtype=np.float32)
        power_text = ""
        if len(selected_powers) > 0:
            power_text = (
                f" | power_mean={float(np.mean(selected_powers)):.4f}"
                f" power_std={float(np.std(selected_powers)):.4f}"
                f" power_min={float(np.min(selected_powers)):.4f}"
                f" power_max={float(np.max(selected_powers)):.4f}"
            )

        validation = None
        saved_best = False
        if episode % max(int(validation_interval), 1) == 0 or episode == num_episodes:
            validation = run_worker_validation(agent, config, validation_scenarios)
            if validation["score"] < best_validation_score:
                best_validation_score = validation["score"]
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
                        "action_mode": "masked_sensor_plus_power",
                        "reward_mode": config.get("worker_reward_mode", "current"),
                        "worker_power_mode": config.get("worker_power_mode", "learned_beta"),
                        "worker_context_mode": config.get("worker_context_mode", "random_feasible_context"),
                        "architecture_variant": f"worker_{config.get('worker_power_mode', 'learned_beta')}",
                        "training_step": int(episode),
                        "validation_metrics": validation,
                        "scenario_suite": scenario_suite_metadata(validation_scenarios),
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
                "train_invalid": int(episode_invalid),
                "train_wasted": int(episode_wasted),
                "train_bh_energy": avg_backhaul_energy,
                "power_mean": float(np.mean(selected_powers)) if len(selected_powers) > 0 else 0.0,
                "power_std": float(np.std(selected_powers)) if len(selected_powers) > 0 else 0.0,
                "power_min": float(np.min(selected_powers)) if len(selected_powers) > 0 else 0.0,
                "power_max": float(np.max(selected_powers)) if len(selected_powers) > 0 else 0.0,
                "val_reward": None if validation is None else validation["avg_reward"],
                "val_aodt": None if validation is None else validation["avg_aodt"],
                "val_invalid": None if validation is None else validation["invalid"],
                "val_wasted": None if validation is None else validation["wasted"],
                "val_score": None if validation is None else validation["score"],
                "update_loss": None if not last_update_stats else last_update_stats["loss"],
                "update_actor_loss": None if not last_update_stats else last_update_stats["actor_loss"],
                "update_critic_loss": None if not last_update_stats else last_update_stats["critic_loss"],
                "update_entropy": None if not last_update_stats else last_update_stats["entropy"],
                "update_approx_kl": None if not last_update_stats else last_update_stats["approx_kl"],
                "update_clip_fraction": None if not last_update_stats else last_update_stats["clip_fraction"],
                "update_explained_variance": None if not last_update_stats else last_update_stats["explained_variance"],
                "update_grad_norm": None if not last_update_stats else last_update_stats["grad_norm"],
                "saved_best": saved_best,
            }
        )

        val_text = " | val_skipped=True"
        if validation is not None:
            val_text = (
                f" | val_aodt={validation['avg_aodt']:.4f} | "
                f"val_invalid={validation['invalid']:.2f} | "
                f"val_wasted={validation['wasted']:.2f}"
            )
            validation_rows.append(
                {
                    "episode": int(episode),
                    "validation_score": float(validation["score"]),
                    "avg_reward": float(validation["avg_reward"]),
                    "avg_aodt": float(validation["avg_aodt"]),
                    "invalid": float(validation["invalid"]),
                    "wasted": float(validation["wasted"]),
                }
            )

        print(
            f"Episode {episode:04d} | "
            f"train_reward={avg_reward:.4f} | "
            f"train_aodt={avg_aodt:.4f} | "
            f"train_invalid={episode_invalid} | "
            f"train_wasted={episode_wasted} | "
            f"train_bh_energy={avg_backhaul_energy:.4f} | "
            f"{val_text}"
            f"{power_text}"
            f"{loss_text}"
            f"{best_text}"
        )

    print("-" * 72)
    print("Saved best worker model:", save_path)
    print("Best validation score:", f"{best_validation_score:.4f}")

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
        "--power-mode",
        choices=["learned_beta", "fixed_max", "fixed_mid"],
        default=None,
    )
    parser.add_argument(
        "--context-mode",
        choices=["fixed_context", "random_feasible_context", "heuristic_context"],
        default=None,
    )
    parser.add_argument(
        "--worker-reward-mode",
        choices=["current"],
        default=None,
    )
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--history-output-path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = dict(CONFIG)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.power_mode is not None:
        config["worker_power_mode"] = args.power_mode
    if args.context_mode is not None:
        config["worker_context_mode"] = args.context_mode
    if args.worker_reward_mode is not None:
        config["worker_reward_mode"] = args.worker_reward_mode

    train_worker(
        config=config,
        num_episodes=args.episodes,
        rollout_size=args.rollout_size,
        save_path=args.save_path,
        fixed_scenario=not args.randomize_scenarios,
        validation_interval=args.validation_interval,
        history_output_path=args.history_output_path,
    )


if __name__ == "__main__":
    main()
