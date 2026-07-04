from pathlib import Path

import numpy as np

from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.base_env import BaseUAVAoDTEnv


def grid_points_from_config(config=None):
    config = dict(CONFIG if config is None else config)
    area_size = float(config["area_size"])
    grid_size = int(config.get("manager_grid_size", 4))
    coords = np.linspace(0.0, area_size, grid_size, dtype=np.float32)
    points = []
    for x in coords:
        for y in coords:
            points.append([x, y])
    return np.asarray(points, dtype=np.float32)


def make_scenario_suite(config=None, count=8, split="validation"):
    config = dict(CONFIG if config is None else config)
    offsets = config.get("scenario_seed_offsets", {})
    split_offset = int(offsets.get(split, 0))
    base_seed = int(config["seed"]) + split_offset
    env = BaseUAVAoDTEnv(config=config)
    scenarios = []
    for idx in range(count):
        scenario = env.make_scenario_snapshot(seed=base_seed + idx)
        scenario["suite_split"] = split
        scenario["suite_index"] = idx
        scenarios.append(scenario)
    return scenarios


def nearest_grid_indices(grid_points, positions):
    distances = np.linalg.norm(
        positions[:, None, :] - grid_points[None, :, :],
        axis=2,
    )
    return np.argmin(distances, axis=1).astype(int)


def greedy_feasible_dt_assignment(base_env, uav_positions):
    entity_centroids = np.zeros((base_env.E, 2), dtype=np.float32)
    for entity_id in range(base_env.E):
        sensor_ids = np.where(base_env.sensor_entity == entity_id)[0]
        entity_centroids[entity_id] = np.mean(base_env.sensor_positions[sensor_ids], axis=0)

    remaining_capacity = base_env.uav_storage_capacity.astype(np.float32).copy()
    dt_hosts = np.full(base_env.E, -1, dtype=int)
    ordered_entities = np.argsort(-base_env.dt_storage)

    for entity_id in ordered_entities:
        candidate_order = np.argsort(
            np.linalg.norm(uav_positions - entity_centroids[entity_id], axis=1)
        )
        chosen = None
        for uav_id in candidate_order:
            if remaining_capacity[uav_id] + 1e-9 >= base_env.dt_storage[entity_id]:
                chosen = int(uav_id)
                break
        if chosen is None:
            feasible = base_env.get_feasible_dt_assignments()
            return feasible[0].copy()
        dt_hosts[entity_id] = chosen
        remaining_capacity[chosen] -= float(base_env.dt_storage[entity_id])

    return dt_hosts


def make_scenario_aware_static_action(base_env, config=None):
    config = dict(CONFIG if config is None else config)
    grid_points = grid_points_from_config(config)
    entity_centroids = np.zeros((base_env.E, 2), dtype=np.float32)
    for entity_id in range(base_env.E):
        sensor_ids = np.where(base_env.sensor_entity == entity_id)[0]
        entity_centroids[entity_id] = np.mean(base_env.sensor_positions[sensor_ids], axis=0)

    chosen_grid_indices = []
    chosen_positions = []
    used_grid_indices = set()

    for centroid in entity_centroids[: base_env.M]:
        distances = np.linalg.norm(grid_points - centroid[None, :], axis=1)
        for candidate in np.argsort(distances):
            candidate = int(candidate)
            if candidate not in used_grid_indices:
                chosen_grid_indices.append(candidate)
                chosen_positions.append(grid_points[candidate])
                used_grid_indices.add(candidate)
                break

    while len(chosen_grid_indices) < base_env.M:
        fallback = len(chosen_grid_indices)
        chosen_grid_indices.append(fallback)
        chosen_positions.append(grid_points[fallback])
        used_grid_indices.add(fallback)

    chosen_positions = np.asarray(chosen_positions, dtype=np.float32)
    dt_hosts = greedy_feasible_dt_assignment(base_env, chosen_positions)
    backhaul_mid = 0.5 * (
        float(config.get("backhaul_power_min", config["backhaul_power"]))
        + float(config.get("backhaul_power_max", config["backhaul_power"]))
    )

    return {
        "uav_grid_indices": np.asarray(chosen_grid_indices, dtype=int),
        "dt_assignment_index": None,
        "dt_hosts": dt_hosts.astype(int),
        "backhaul_powers": np.ones(base_env.M, dtype=np.float32) * backhaul_mid,
    }


def sample_worker_context(base_env, mode="random_feasible_context", rng=None, config=None):
    config = dict(CONFIG if config is None else config)
    rng = base_env.rng if rng is None else rng
    grid_points = grid_points_from_config(config)

    if mode == "fixed_context":
        return {
            "uav_positions": base_env.uav_positions.copy(),
            "dt_hosts": base_env.dt_hosts.copy(),
            "backhaul_powers": base_env.backhaul_powers.copy(),
        }

    if mode == "random_feasible_context":
        grid_indices = rng.integers(low=0, high=len(grid_points), size=base_env.M)
        return {
            "uav_positions": grid_points[grid_indices].copy(),
            "dt_hosts": base_env.sample_random_feasible_dt_assignment(rng=rng),
            "backhaul_powers": rng.uniform(
                low=config.get("backhaul_power_min", config["backhaul_power"]),
                high=config.get("backhaul_power_max", config["backhaul_power"]),
                size=base_env.M,
            ).astype(np.float32),
        }

    if mode == "heuristic_context":
        static_action = make_scenario_aware_static_action(base_env, config=config)
        return {
            "uav_positions": grid_points[static_action["uav_grid_indices"]].copy(),
            "dt_hosts": static_action["dt_hosts"].copy(),
            "backhaul_powers": static_action["backhaul_powers"].copy(),
        }

    raise ValueError(f"Unknown worker context mode: {mode}")


def scenario_suite_metadata(scenarios):
    return {
        "count": len(scenarios),
        "scenario_seeds": [int(s["scenario_seed"]) for s in scenarios],
        "split": scenarios[0].get("suite_split") if scenarios else None,
        "feasible_assignment_hashes": [s.get("feasible_assignment_hash") for s in scenarios],
    }
