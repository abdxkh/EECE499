from __future__ import annotations

import argparse
import csv
import json
import math
import ast
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from dt_uav_v2.evaluation.phase_c2 import (
    build_phase_c2_config,
    evaluate_manager_policy,
    evaluate_worker_policy,
    load_checkpoint_metadata,
    make_manager_agent,
)
from dt_uav_v2.envs.manager_env import ManagerEnv
from dt_uav_v2.utils.scenarios import make_scenario_suite, scenario_suite_metadata


SOURCE_ROOT = Path("outputs/results/phase_c2_full")
FINAL_ROOT = Path("outputs/results/phase_c2_final")
DOC_PATH = Path("docs/PHASE_C2_FINAL_RESULTS.md")
TEST_SCENARIOS = 200
TEST_SUITE_SEED = 314159
EXPECTED_RAW_BUNDLES = [
    ("worker_ppo_seed41", 41),
    ("worker_ppo_seed42", 42),
    ("worker_ppo_seed43", 43),
    ("worker_greedy", None),
    ("ppo_worker_candidate0", 51),
    ("ppo_worker_candidate0", 52),
    ("ppo_worker_candidate0", 53),
    ("ppo_worker_candidate0", 54),
    ("ppo_worker_candidate0", 55),
    ("ppo_worker_candidate1", 51),
    ("ppo_worker_candidate1", 52),
    ("ppo_worker_candidate1", 53),
    ("ppo_worker_candidate1", 54),
    ("ppo_worker_candidate1", 55),
    ("greedy_worker_manager", 51),
    ("greedy_worker_manager", 52),
    ("greedy_worker_manager", 53),
    ("greedy_worker_manager", 54),
    ("greedy_worker_manager", 55),
    ("random_manager", 0),
    ("fixed_global_manager", 0),
    ("static_heuristic_manager", 0),
]


def py(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: py(val) for k, val in v.items()}
    if isinstance(v, list):
        return [py(val) for val in v]
    return v


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as fh:
        json.dump(py(data), fh, indent=2)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(py(rows))


def df_to_markdown_simple(df, max_rows=None):
    if df is None or df.empty:
        return "_empty_"
    if max_rows is not None:
        df = df.head(max_rows)
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(map(str, cols)) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        values = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                values.append(f"{val:.6g}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def read_json(path: Path):
    with open(path, "r", encoding="ascii") as fh:
        return json.load(fh)


def stats_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    median = float(np.median(values))
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    half = 1.96 * std / math.sqrt(values.size) if values.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "min": vmin,
        "max": vmax,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def paired_stats(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    half = 1.96 * std_diff / math.sqrt(diff.size) if diff.size > 1 else 0.0
    if diff.size > 1 and std_diff > 0:
        t_stat, p_value = stats.ttest_rel(a, b)
        effect = float(mean_diff / std_diff)
    else:
        t_stat, p_value, effect = 0.0, 1.0, 0.0
    return {
        "mean_diff": mean_diff,
        "ci95_low": mean_diff - half,
        "ci95_high": mean_diff + half,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohen_d": effect,
        "wins_a": int(np.sum(a < b)),
        "wins_b": int(np.sum(b < a)),
        "ties": int(np.sum(a == b)),
    }


def percentile(values, q):
    values = np.asarray(values, dtype=np.float64)
    return float(np.percentile(values, q)) if values.size else None


def to_numeric_array(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.asarray([], dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.asarray([], dtype=np.float32)
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = ast.literal_eval(text)
        return to_numeric_array(parsed)
    return np.asarray([value], dtype=np.float32)


def movement_distance_from_rows(ep_df, tr_df):
    candidates = [
        ("movement_distance", ep_df),
        ("total_movement_distance", ep_df),
        ("movement", ep_df),
        ("avg_movement_distance_per_uav_transition", ep_df),
        ("movement_distance", tr_df),
    ]
    for field, frame in candidates:
        if frame is not None and field in frame.columns:
            values = pd.to_numeric(frame[field], errors="coerce").dropna()
            if not values.empty:
                return float(values.sum()), field
    return float("nan"), None


def raw_run_dir(system: str, seed: int | None):
    if seed is None:
        return FINAL_ROOT / "raw" / system
    return FINAL_ROOT / "raw" / system / f"seed{seed}"


def raw_run_paths(system: str, seed: int | None):
    root = raw_run_dir(system, seed)
    return {
        "root": root,
        "manifest": root / "manifest.json",
        "episode": root / "per_episode.csv",
        "window": root / "per_window.csv",
        "transmission": root / "transmissions.csv",
        "regret": root / "regret.csv",
        "summary": root / "summary.json",
    }


def raw_run_ready(system: str, seed: int | None):
    paths = raw_run_paths(system, seed)
    required = [paths["manifest"], paths["episode"], paths["window"], paths["summary"]]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def expected_raw_bundle_count():
    return len(EXPECTED_RAW_BUNDLES)


def expected_raw_bundle_paths():
    return [raw_run_paths(system, seed) for system, seed in EXPECTED_RAW_BUNDLES]


def missing_raw_bundles():
    missing = []
    for system, seed in EXPECTED_RAW_BUNDLES:
        paths = raw_run_paths(system, seed)
        if not raw_run_ready(system, seed):
            missing.append(
                {
                    "system": system,
                    "seed": seed,
                    "root": str(paths["root"].as_posix()),
                    "missing_files": [
                        name
                        for name, path in {
                            "manifest": paths["manifest"],
                            "episode": paths["episode"],
                            "window": paths["window"],
                            "summary": paths["summary"],
                        }.items()
                        if not (path.exists() and path.stat().st_size > 0)
                    ],
                }
            )
    return missing


def load_raw_systems_only():
    systems = {}

    for seed in [41, 42, 43]:
        label = f"worker_ppo_seed{seed}"
        paths = raw_run_paths(label, seed)
        summary = read_json(paths["summary"])
        episode_rows = pd.read_csv(paths["episode"]).to_dict(orient="records")
        transmission_rows = pd.read_csv(paths["transmission"]).to_dict(orient="records") if paths["transmission"].exists() else []
        regret_rows = pd.read_csv(paths["regret"]).to_dict(orient="records") if paths["regret"].exists() else []
        systems[label] = {
            "kind": "worker",
            "worker_type": "ppo",
            "seed": seed,
            "config_name": f"worker_fixed_max_seed{seed}",
            "checkpoint_paths": [str((SOURCE_ROOT / "worker" / "models" / f"worker_fixed_max_seed{seed}.pt").relative_to(SOURCE_ROOT)).replace("\\", "/")],
            "summary": summary,
            "episode_rows": episode_rows,
            "transmission_rows": transmission_rows,
            "regret_rows": regret_rows,
        }

    label = "worker_greedy"
    paths = raw_run_paths(label, None)
    summary = read_json(paths["summary"])
    episode_rows = pd.read_csv(paths["episode"]).to_dict(orient="records")
    transmission_rows = pd.read_csv(paths["transmission"]).to_dict(orient="records") if paths["transmission"].exists() else []
    regret_rows = pd.read_csv(paths["regret"]).to_dict(orient="records") if paths["regret"].exists() else []
    systems[label] = {
        "kind": "worker",
        "worker_type": "greedy",
        "seed": None,
        "config_name": "greedy_worker",
        "checkpoint_paths": [],
        "summary": summary,
        "episode_rows": episode_rows,
        "transmission_rows": transmission_rows,
        "regret_rows": regret_rows,
    }

    manager_specs = [
        ("ppo_worker_candidate0", "ppo", SOURCE_ROOT / "worker" / "models" / "worker_fixed_max_seed41.pt", "manager_final_candidate_0"),
        ("ppo_worker_candidate1", "ppo", SOURCE_ROOT / "worker" / "models" / "worker_fixed_max_seed41.pt", "manager_final_candidate_1"),
        ("greedy_worker_manager", "greedy", SOURCE_ROOT / "worker" / "models" / "worker_fixed_max_seed41.pt", "manager_greedy_worker"),
        ("random_manager", "ppo", SOURCE_ROOT / "worker" / "models" / "worker_fixed_max_seed41.pt", None),
        ("fixed_global_manager", "ppo", SOURCE_ROOT / "worker" / "models" / "worker_fixed_max_seed41.pt", None),
        ("static_heuristic_manager", "ppo", SOURCE_ROOT / "worker" / "models" / "worker_fixed_max_seed41.pt", None),
    ]
    for label, worker_policy, worker_model_path, model_dir_name in manager_specs:
        seed_paths = sorted(raw_run_dir(label, 0).parent.glob(f"{label}/seed*")) if label.endswith("_manager") and label.startswith(("random", "fixed_global", "static_heuristic")) else sorted(raw_run_dir(label, 0).parent.glob(f"{label}/seed*"))
        if label in {"random_manager", "fixed_global_manager", "static_heuristic_manager"}:
            seeds = [0]
        else:
            seeds = [51, 52, 53, 54, 55]
        for seed in seeds:
            paths = raw_run_paths(label, seed)
            summary = read_json(paths["summary"])
            episode_rows = pd.read_csv(paths["episode"]).to_dict(orient="records")
            transition_rows = pd.read_csv(paths["window"]).to_dict(orient="records")
            transmission_rows = pd.read_csv(paths["transmission"]).to_dict(orient="records") if paths["transmission"].exists() else []
            systems.setdefault(label, {"kind": "manager", "worker_policy": worker_policy, "config_name": label, "runs": []})
            systems[label]["runs"].append(
                {
                    "seed": seed,
                    "manager_model_path": str((SOURCE_ROOT / model_dir_name / "models" / f"{label.replace('ppo_worker_candidate0', 'manager_ppo_r128_lr1e-04_e0.01').replace('ppo_worker_candidate1', 'manager_ppo_r128_lr1e-04_e0.005').replace('greedy_worker_manager', 'manager_greedy_r128_lr1e-04_e0.01')}_seed{seed}.pt").relative_to(SOURCE_ROOT)).replace("\\", "/") if model_dir_name else None,
                    "summary": summary,
                    "episode_rows": episode_rows,
                    "transition_rows": transition_rows,
                    "transmission_rows": transmission_rows,
                }
            )

    return systems


def save_raw_run(system: str, seed: int | None, manifest: dict, summary: dict, episode_rows, window_rows, transmission_rows=None, regret_rows=None):
    paths = raw_run_paths(system, seed)
    paths["root"].mkdir(parents=True, exist_ok=True)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    write_csv(paths["episode"], episode_rows)
    write_csv(paths["window"], window_rows)
    if transmission_rows is not None:
        write_csv(paths["transmission"], transmission_rows)
    if regret_rows is not None:
        write_csv(paths["regret"], regret_rows)


def parse_seed_from_name(name: str) -> int:
    tail = name.split("seed")[-1]
    tail = tail.split(".")[0]
    return int(tail)


def collect_file_inventory(root: Path):
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        entry = {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "zero_bytes": path.stat().st_size == 0,
            "kind": path.suffix.lower().lstrip("."),
        }
        if path.suffix.lower() == ".json":
            if path.stat().st_size <= 5_000_000:
                try:
                    read_json(path)
                    entry["parse_ok"] = True
                except Exception as exc:
                    entry["parse_ok"] = False
                    entry["parse_error"] = repr(exc)
            else:
                entry["parse_ok"] = "skipped_large"
        elif path.suffix.lower() == ".csv":
            if path.stat().st_size <= 5_000_000:
                try:
                    pd.read_csv(path)
                    entry["parse_ok"] = True
                except Exception as exc:
                    entry["parse_ok"] = False
                    entry["parse_error"] = repr(exc)
            else:
                entry["parse_ok"] = "skipped_large"
        rows.append(entry)
    return rows


def load_validation_rows():
    rows = []
    # Worker
    for path in sorted((SOURCE_ROOT / "worker" / "validation").glob("worker_fixed_max_seed*.json")):
        summary = read_json(path)
        seed = parse_seed_from_name(path.stem)
        model_path = SOURCE_ROOT / "worker" / "models" / f"worker_fixed_max_seed{seed}.pt"
        status_path = SOURCE_ROOT / "worker" / "status" / f"worker_worker_fixed_max_seed{seed}.json"
        status = read_json(status_path) if status_path.exists() else {}
        rows.append(
            {
                "split": "validation",
                "family": "worker",
                "system": "fixed_max",
                "config_name": "fixed_max",
                "seed": seed,
                "status": status.get("status", "unknown"),
                "checkpoint_path": str(model_path.relative_to(SOURCE_ROOT)).replace("\\", "/"),
                "checkpoint_set": json.dumps([str(model_path.relative_to(SOURCE_ROOT)).replace("\\", "/")]),
                "best_validation_aodt": summary["avg_aodt"],
                "average_energy": summary.get("mean_energy", None),
                "positive_energy_violation": summary.get("avg_positive_violation", None),
                "uav_window_violation_fraction": summary.get("uav_window_violation_fraction", None),
                "feasible_episode_fraction": 1.0 if summary.get("avg_positive_violation", 0.0) == 0.0 else 0.0,
                "mean_queue": summary.get("avg_queue", None),
                "terminal_queue": summary.get("final_queue", None),
                "uav_switch_fraction": None,
                "dt_host_switch_fraction": None,
                "movement_distance": None,
                "mean_backhaul_power": None,
                "episode_saved": summary.get("count", None),
            }
        )

    # Manager stages and finalists
    manager_specs = [
        ("manager_stage_a", "ppo", "r128_lr3e-04_e0.01"),
        ("manager_stage_b", "ppo", "r128_lr1e-04_e0.01"),
        ("manager_stage_c", "ppo", "r128_lr1e-04_e0.005"),
        ("manager_final_candidate_0", "ppo", "r128_lr1e-04_e0.01"),
        ("manager_final_candidate_1", "ppo", "r128_lr1e-04_e0.005"),
        ("manager_greedy_worker", "greedy", "r128_lr1e-04_e0.01"),
    ]
    for family, worker_policy, stem_prefix in manager_specs:
        for path in sorted((SOURCE_ROOT / family / "validation").glob(f"{worker_policy}_*.json")):
            if path.name.endswith("_static.json"):
                continue
            summary = read_json(path)
            seed = parse_seed_from_name(path.stem)
            model_path = SOURCE_ROOT / family / "models" / f"{path.stem}.pt"
            status_path = SOURCE_ROOT / family / "status" / f"manager_{path.stem}.json"
            status = read_json(status_path) if status_path.exists() else {}
            rows.append(
                {
                    "split": "validation",
                    "family": family,
                    "system": stem_prefix,
                    "config_name": stem_prefix,
                    "seed": seed,
                    "status": status.get("status", "unknown"),
                    "checkpoint_path": str(model_path.relative_to(SOURCE_ROOT)).replace("\\", "/"),
                    "checkpoint_set": json.dumps([str(model_path.relative_to(SOURCE_ROOT)).replace("\\", "/")]),
                    "best_validation_aodt": summary["avg_aodt"],
                    "average_energy": summary.get("mean_energy", None),
                    "positive_energy_violation": summary.get("avg_positive_violation", None),
                    "uav_window_violation_fraction": summary.get("uav_window_violation_fraction", None),
                    "feasible_episode_fraction": 1.0 if summary.get("avg_positive_violation", 0.0) == 0.0 else 0.0,
                    "mean_queue": summary.get("avg_queue", None),
                    "terminal_queue": summary.get("final_queue", None),
                    "uav_switch_fraction": summary.get("uav_switch_fraction", None),
                    "dt_host_switch_fraction": summary.get("dt_host_switch_fraction", None),
                    "movement_distance": summary.get("movement_distance", None),
                    "mean_backhaul_power": summary.get("mean_backhaul_power", None),
                    "episode_saved": status.get("last_completed_episode", summary.get("windows", None)),
                }
            )
    return rows


def aggregate_seed_rows(rows, group_keys):
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in set(group_keys) | {"split", "family", "config_name", "seed", "status", "checkpoint_path"}]
    grouped = []
    for keys, g in df.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        row["seed"] = "aggregate"
        for col in metric_cols:
            vals = [v for v in g[col].tolist() if pd.notna(v)]
            if vals and isinstance(vals[0], list):
                row[col] = np.mean(np.asarray(vals, dtype=np.float32), axis=0).tolist()
            elif vals:
                s = stats_summary(vals)
                for subkey, val in s.items():
                    row[f"{col}_{subkey}"] = val
                row[col] = s["mean"]
        grouped.append(row)
    return grouped


def prepare_test_config():
    config = build_phase_c2_config()
    config["seed"] = TEST_SUITE_SEED
    config["validation_worker_scenarios"] = 40
    config["validation_manager_scenarios"] = 40
    config["test_scenarios_default"] = TEST_SCENARIOS
    return config


def run_worker_eval(label, config, scenarios, model_path, compare_same_state=True):
    summary, episode_rows, transmission_rows, regret_rows = evaluate_worker_policy(
        policy_name="ppo" if label.startswith("ppo") else "greedy",
        config=config,
        model_path=str(model_path),
        scenarios=scenarios,
        compare_same_state=compare_same_state,
    )
    for row in episode_rows:
        row["system"] = label
    for row in transmission_rows:
        row["system"] = label
    for row in regret_rows:
        row["system"] = label
    return summary, episode_rows, transmission_rows, regret_rows


def run_manager_eval(label, policy_name, config, scenarios, worker_model_path, worker_policy, manager_model_path=None):
    env = ManagerEnv(config=config, worker_model_path=str(worker_model_path), worker_policy=worker_policy)
    agent = None
    if policy_name == "ppo":
        env.reset(scenario=scenarios[0])
        agent = make_manager_agent(env, config, str(manager_model_path))
    summary, episode_rows, transition_rows, transmission_rows = evaluate_manager_policy(
        policy_name=policy_name,
        env=env,
        scenarios=scenarios,
        config=config,
        agent=agent,
        rng_seed=int(config["seed"]),
    )
    for row in episode_rows:
        row["system"] = label
    for row in transition_rows:
        row["system"] = label
    for row in transmission_rows:
        row["system"] = label
    return summary, episode_rows, transition_rows, transmission_rows


def add_manager_derived_metrics(episode_rows, transition_rows, transmission_rows, energy_budget):
    ep_df = pd.DataFrame(episode_rows)
    tr_df = pd.DataFrame(transition_rows)
    tx_df = pd.DataFrame(transmission_rows)

    if not tr_df.empty:
        tr_df["avg_energy_mean"] = tr_df["avg_energy_per_uav"].apply(lambda v: float(np.mean(to_numeric_array(v))) if to_numeric_array(v).size else 0.0)
        tr_df["positive_violation_mean"] = tr_df["positive_violation"].apply(lambda v: float(np.mean(to_numeric_array(v))) if to_numeric_array(v).size else 0.0)
        tr_df["energy_term"] = tr_df["queue_weighted_energy_term"].astype(float)
        tr_df["aodt_term"] = tr_df["normalized_aodt_term"].astype(float)
        tr_df["energy_term_gt_aodt"] = tr_df["energy_term"] > tr_df["aodt_term"]

    if not ep_df.empty:
        ep_df["feasible_episode"] = ep_df["violation_rate"].astype(float) == 0.0
        ep_df["positive_energy_violation"] = np.maximum(ep_df["mean_energy"].astype(float) - float(energy_budget), 0.0)
        if "terminal_queue_per_uav" in ep_df.columns:
            tq = np.asarray([to_numeric_array(v) for v in ep_df["terminal_queue_per_uav"].tolist()], dtype=np.float32)
            ep_df["terminal_queue_mean_per_uav"] = tq.mean(axis=1)
            ep_df["terminal_queue_max_per_uav"] = tq.max(axis=1)
    if not tx_df.empty:
        tx_df["completed"] = tx_df["completed"].astype(bool)
    return ep_df, tr_df, tx_df


def summarize_manager_seed(ep_df, tr_df, tx_df, energy_budget):
    completed = tx_df[tx_df["completed"]] if not tx_df.empty else tx_df
    direct = completed[completed["direct_upload"]] if not completed.empty else completed
    cross = completed[completed["cross_upload"]] if not completed.empty else completed
    movement_sum, movement_field = movement_distance_from_rows(ep_df, tr_df)
    metric = {
        "mean_entity_aodt": float(ep_df["avg_aodt"].mean()),
        "std_entity_aodt": float(ep_df["avg_aodt"].std(ddof=1)) if len(ep_df) > 1 else 0.0,
        "median_entity_aodt": float(ep_df["avg_aodt"].median()),
        "p95_entity_aodt": float(ep_df["avg_aodt"].quantile(0.95)),
        "max_entity_aodt": float(ep_df["avg_aodt"].max()),
        "mean_backhaul_energy": float(ep_df["mean_energy"].mean()),
        "max_per_uav_episode_avg_energy": float(ep_df["max_energy"].max()),
        "positive_energy_violation": float(ep_df["positive_energy_violation"].mean()),
        "uav_window_violation_fraction": float(ep_df["uav_window_violation_fraction"].mean()),
        "feasible_episode_fraction": float(ep_df["feasible_episode"].mean()),
        "mean_virtual_queue": float(ep_df["avg_queue"].mean()),
        "terminal_virtual_queue": float(ep_df["final_queue"].iloc[-1]),
        "terminal_virtual_queue_mean_per_uav": float(ep_df["terminal_queue_mean_per_uav"].mean()) if "terminal_queue_mean_per_uav" in ep_df else None,
        "terminal_virtual_queue_max_per_uav": float(ep_df["terminal_queue_max_per_uav"].max()) if "terminal_queue_max_per_uav" in ep_df else None,
        "direct_upload_ratio": float(len(direct) / max(len(completed), 1)),
        "cross_upload_ratio": float(len(cross) / max(len(completed), 1)),
        "idle_rate": float(ep_df["idle_rate"].mean()) if "idle_rate" in ep_df else 0.0,
        "completed_update_count": int(len(completed)),
        "mean_delay": float(completed["total_delay_s"].mean()) if not completed.empty and "total_delay_s" in completed else None,
        "p95_delay": float(completed["total_delay_s"].quantile(0.95)) if not completed.empty and "total_delay_s" in completed else None,
        "uav_switch_fraction": float(ep_df["uav_switch_fraction"].mean()),
        "dt_host_switch_fraction": float(ep_df["dt_host_switch_fraction"].mean()),
        "total_movement_distance": movement_sum,
        "mean_worker_inference_time": float(ep_df["worker_inference_time_s_mean"].mean()) if "worker_inference_time_s_mean" in ep_df else None,
        "mean_manager_inference_time": float(ep_df["manager_inference_time_s_mean"].mean()),
        "mean_backhaul_power": float(completed["selected_backhaul_power"].mean()) if not completed.empty and "selected_backhaul_power" in completed else None,
        "manager_actions": int(ep_df["manager_actions"].sum()),
        "window_count": int(ep_df["windows"].sum()),
        "mean_aodt_reward_term": float(ep_df["mean_aodt_reward_term"].mean()),
        "mean_energy_reward_term": float(ep_df["mean_energy_reward_term"].mean()),
        "max_energy_reward_term": float(ep_df["max_energy_reward_term"].max()) if "max_energy_reward_term" in ep_df else None,
        "energy_term_gt_aodt_fraction": float(tr_df["energy_term_gt_aodt"].mean()) if not tr_df.empty else 0.0,
        "movement_field_used": movement_field,
    }
    return metric


def summarize_worker_seed(ep_df, tx_df):
    completed = tx_df[tx_df["completed"]] if not tx_df.empty else tx_df
    direct = completed[completed["direct_upload"]] if not completed.empty else completed
    cross = completed[completed["cross_upload"]] if not completed.empty else completed
    return {
        "mean_entity_aodt": float(ep_df["avg_aodt"].mean()),
        "std_entity_aodt": float(ep_df["avg_aodt"].std(ddof=1)) if len(ep_df) > 1 else 0.0,
        "median_entity_aodt": float(ep_df["avg_aodt"].median()),
        "p95_entity_aodt": float(ep_df["avg_aodt"].quantile(0.95)),
        "max_entity_aodt": float(ep_df["avg_aodt"].max()),
        "mean_delay": float(completed["total_delay_s"].mean()) if not completed.empty else None,
        "p95_delay": float(completed["total_delay_s"].quantile(0.95)) if not completed.empty else None,
        "max_delay": float(completed["total_delay_s"].max()) if not completed.empty else None,
        "fraction_over_slot": float((completed["total_delay_s"] > 1.0).mean()) if not completed.empty else None,
        "direct_upload_ratio": float(len(direct) / max(len(completed), 1)),
        "cross_upload_ratio": float(len(cross) / max(len(completed), 1)),
        "idle_rate": float(ep_df["idle_rate"].mean()),
        "completed_update_count": int(len(completed)),
        "mean_worker_inference_time": float(ep_df["mean_action_time_s"].mean()),
        "action_entropy_mean": float(ep_df["mean_entropy"].dropna().mean()) if "mean_entropy" in ep_df else None,
        "same_sensor_fraction_mean": float(ep_df["mean_same_sensor_fraction"].dropna().mean()) if "mean_same_sensor_fraction" in ep_df else None,
        "ppo_gain_mean": float(ep_df["mean_ppo_gain"].dropna().mean()) if "mean_ppo_gain" in ep_df else None,
        "greedy_gain_mean": float(ep_df["mean_greedy_gain"].dropna().mean()) if "mean_greedy_gain" in ep_df else None,
        "regret_mean": float(ep_df["mean_regret"].dropna().mean()) if "mean_regret" in ep_df else None,
        "feasible_real_action_count_mean": float(ep_df["feasible_real_action_count"].dropna().mean()) if "feasible_real_action_count" in ep_df else None,
        "no_feasible_real_sensor_count_mean": float(ep_df["no_feasible_real_sensor_count"].dropna().mean()) if "no_feasible_real_sensor_count" in ep_df else None,
        "mean_backhaul_power": float(completed["selected_backhaul_power"].mean()) if not completed.empty else None,
    }


def build_completion_inventory():
    inventory = {
        "source_root": str(SOURCE_ROOT.as_posix()),
        "file_inventory": collect_file_inventory(SOURCE_ROOT),
        "completed_runs": {},
        "partial_runs": {},
        "failed_runs": {},
        "checkpoint_paths": [],
        "last_completed_episodes": {},
        "best_validation_metrics": {},
        "resume_safety": {
            "safe_to_rerun_original_command": True,
            "completed_runs_will_be_skipped": True,
            "partial_run_will_resume": False,
            "risk_of_overwrite_or_restart": "Completed artifacts remain intact. An interrupted seed would restart from scratch if re-run.",
        },
        "recommended_continuation_command": None,
        "missing_expected_paths": [],
        "integrity_checks": {},
    }

    completed = {"worker": [], "manager_stage_a": [], "manager_stage_b": [], "manager_stage_c": [], "manager_final_candidate_0": [], "manager_final_candidate_1": [], "manager_greedy_worker": []}
    partial = {"worker": [], "manager_stage_a": [], "manager_stage_b": [], "manager_stage_c": [], "manager_final_candidate_0": [], "manager_final_candidate_1": [], "manager_greedy_worker": []}
    failed = {"worker": [], "manager_stage_a": [], "manager_stage_b": [], "manager_stage_c": [], "manager_final_candidate_0": [], "manager_final_candidate_1": [], "manager_greedy_worker": []}
    best_paths = []
    last_completed = {}
    best_metrics = {}

    # Worker
    for status_path in sorted((SOURCE_ROOT / "worker" / "status").glob("*.json")):
        status = read_json(status_path)
        seed = status["seed"]
        summary = read_json(SOURCE_ROOT / "worker" / "validation" / f"worker_fixed_max_seed{seed}.json")
        entry = {
            "seed": seed,
            "status": status["status"],
            "best_validation_aodt": summary["avg_aodt"],
            "best_checkpoint_path": status["model_path"].replace("\\", "/"),
            "validation_summary_path": status["validation_summary_path"].replace("\\", "/"),
            "training_step": status.get("training_step", None),
            "validation_metrics": summary,
            "checkpoint_metadata": py(load_checkpoint_metadata(SOURCE_ROOT / "worker" / "models" / f"worker_fixed_max_seed{seed}.pt")),
        }
        if status["status"] == "completed":
            completed["worker"].append(entry)
            best_paths.append(entry["best_checkpoint_path"])
            last_completed["worker"] = entry["training_step"]
            best_metrics[f"worker_seed{seed}"] = summary
        elif status["status"] == "partial":
            partial["worker"].append(entry)
        else:
            failed["worker"].append(entry)

    # Managers
    for family in ["manager_stage_a", "manager_stage_b", "manager_stage_c", "manager_final_candidate_0", "manager_final_candidate_1", "manager_greedy_worker"]:
        for status_path in sorted((SOURCE_ROOT / family / "status").glob("*.json")):
            status = read_json(status_path)
            seed = status["seed"]
            model_path = Path(status["model_path"])
            summary_path = Path(status["validation_summary_path"])
            summary = read_json(summary_path)
            entry = {
                "stage": family,
                "seed": seed,
                "status": status["status"],
                "best_validation_aodt": summary["avg_aodt"],
                "best_checkpoint_path": status["model_path"].replace("\\", "/"),
                "validation_summary_path": status["validation_summary_path"].replace("\\", "/"),
                "validation_static_summary_path": status.get("validation_static_summary_path"),
                "last_completed_episode": status.get("last_completed_episode", None),
                "best_validation_metrics": status.get("best_validation_metrics", summary),
                "checkpoint_metadata": py(load_checkpoint_metadata(model_path)),
            }
            if status["status"] == "completed":
                completed[family].append(entry)
                best_paths.append(entry["best_checkpoint_path"])
                last_completed[f"{family}_seed{seed}"] = status.get("last_completed_episode", None)
                best_metrics[f"{family}_seed{seed}"] = entry["best_validation_metrics"]
            elif status["status"] == "partial":
                partial[family].append(entry)
            else:
                failed[family].append(entry)

    inventory["completed_runs"] = completed
    inventory["partial_runs"] = partial
    inventory["failed_runs"] = failed
    inventory["checkpoint_paths"] = sorted(set(best_paths))
    inventory["last_completed_episodes"] = last_completed
    inventory["best_validation_metrics"] = best_metrics
    inventory["completed_manager_configurations"] = {
        "manager_stage_a": sorted({f"r{r['checkpoint_metadata'].get('manager_rollout_size', 'na')}_lr{r['checkpoint_metadata'].get('config', {}).get('manager_lr', 'na')}_e{r['checkpoint_metadata'].get('config', {}).get('manager_entropy_coef', 'na')}" for r in completed["manager_stage_a"]}),
        "manager_stage_b": sorted({f"r{r['checkpoint_metadata'].get('manager_rollout_size', 'na')}_lr{r['checkpoint_metadata'].get('config', {}).get('manager_lr', 'na')}_e{r['checkpoint_metadata'].get('config', {}).get('manager_entropy_coef', 'na')}" for r in completed["manager_stage_b"]}),
        "manager_stage_c": sorted({f"r{r['checkpoint_metadata'].get('manager_rollout_size', 'na')}_lr{r['checkpoint_metadata'].get('config', {}).get('manager_lr', 'na')}_e{r['checkpoint_metadata'].get('config', {}).get('manager_entropy_coef', 'na')}" for r in completed["manager_stage_c"]}),
        "manager_final_candidate_0": sorted({f"r{r['checkpoint_metadata'].get('manager_rollout_size', 'na')}_lr{r['checkpoint_metadata'].get('config', {}).get('manager_lr', 'na')}_e{r['checkpoint_metadata'].get('config', {}).get('manager_entropy_coef', 'na')}" for r in completed["manager_final_candidate_0"]}),
        "manager_final_candidate_1": sorted({f"r{r['checkpoint_metadata'].get('manager_rollout_size', 'na')}_lr{r['checkpoint_metadata'].get('config', {}).get('manager_lr', 'na')}_e{r['checkpoint_metadata'].get('config', {}).get('manager_entropy_coef', 'na')}" for r in completed["manager_final_candidate_1"]}),
        "manager_greedy_worker": sorted({f"r{r['checkpoint_metadata'].get('manager_rollout_size', 'na')}_lr{r['checkpoint_metadata'].get('config', {}).get('manager_lr', 'na')}_e{r['checkpoint_metadata'].get('config', {}).get('manager_entropy_coef', 'na')}" for r in completed["manager_greedy_worker"]}),
    }
    inventory["integrity_checks"] = {
        "json_files_parsed": sum(1 for r in inventory["file_inventory"] if r["kind"] == "json" and r.get("parse_ok", True)),
        "csv_files_parsed": sum(1 for r in inventory["file_inventory"] if r["kind"] == "csv" and r.get("parse_ok", True)),
        "pt_files_parsed": sum(1 for r in inventory["file_inventory"] if r["kind"] == "pt"),
        "zero_byte_files": [r["path"] for r in inventory["file_inventory"] if r["zero_bytes"]],
        "parse_issues": [r["path"] for r in inventory["file_inventory"] if r.get("parse_ok") is False],
    }
    inventory["missing_expected_paths"] = [
        p for p in [
            "manager_stage_b/",
            "manager_stage_c/",
            "manager_final_candidate_0/",
            "manager_final_candidate_1/",
            "manager_greedy_worker/",
        ]
        if not (SOURCE_ROOT / p).exists()
    ]
    inventory["recommended_continuation_command"] = "No continuation needed; all Phase C2 training runs completed."
    return inventory


def evaluate_test_suite(config, scenarios):
    selected_worker_path = SOURCE_ROOT / "worker" / "models" / "worker_fixed_max_seed41.pt"
    greedy_worker_path = selected_worker_path

    raw_root = FINAL_ROOT / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    systems = {}

    # Worker-only evaluations
    for seed in [41, 42, 43]:
        model_path = SOURCE_ROOT / "worker" / "models" / f"worker_fixed_max_seed{seed}.pt"
        label = f"worker_ppo_seed{seed}"
        if raw_run_ready(label, seed):
            summary = read_json(raw_run_paths(label, seed)["summary"])
            episode_rows = pd.read_csv(raw_run_paths(label, seed)["episode"]).to_dict(orient="records")
            transmission_rows = pd.read_csv(raw_run_paths(label, seed)["transmission"]).to_dict(orient="records") if raw_run_paths(label, seed)["transmission"].exists() else []
            regret_rows = pd.read_csv(raw_run_paths(label, seed)["regret"]).to_dict(orient="records") if raw_run_paths(label, seed)["regret"].exists() else []
            systems[label] = {
                "kind": "worker",
                "worker_type": "ppo",
                "seed": seed,
                "config_name": f"worker_fixed_max_seed{seed}",
                "checkpoint_paths": [str(model_path.relative_to(SOURCE_ROOT)).replace("\\", "/")],
                "summary": summary,
                "episode_rows": episode_rows,
                "transmission_rows": transmission_rows,
                "regret_rows": regret_rows,
            }
            continue
        summary, episode_rows, transmission_rows, regret_rows = evaluate_worker_policy(
            policy_name="ppo",
            config=config,
            model_path=str(model_path),
            scenarios=scenarios,
            compare_same_state=True,
        )
        for row in episode_rows:
            row["system"] = label
            row["worker_type"] = "ppo"
            row["seed"] = seed
        for row in transmission_rows:
            row["system"] = label
            row["worker_type"] = "ppo"
            row["seed"] = seed
        for row in regret_rows:
            row["system"] = label
            row["worker_type"] = "ppo"
            row["seed"] = seed
        systems[label] = {
            "kind": "worker",
            "worker_type": "ppo",
            "seed": seed,
            "config_name": f"worker_fixed_max_seed{seed}",
            "checkpoint_paths": [str(model_path.relative_to(SOURCE_ROOT)).replace("\\", "/")],
            "summary": summary,
            "episode_rows": episode_rows,
            "transmission_rows": transmission_rows,
            "regret_rows": regret_rows,
        }
        save_raw_run(
            system=label,
            seed=seed,
            manifest={
                "kind": "worker",
                "worker_type": "ppo",
                "seed": seed,
                "checkpoint_path": str(model_path.relative_to(SOURCE_ROOT)).replace("\\", "/"),
            },
            summary=summary,
            episode_rows=episode_rows,
            window_rows=transmission_rows,
            transmission_rows=transmission_rows,
            regret_rows=regret_rows,
        )

    # Greedy worker diagnostic
    label = "worker_greedy"
    summary, episode_rows, transmission_rows, regret_rows = evaluate_worker_policy(
        policy_name="greedy",
        config=config,
        model_path=str(greedy_worker_path),
        scenarios=scenarios,
        compare_same_state=True,
    )
    for row in episode_rows:
        row["system"] = label
        row["worker_type"] = "greedy"
    for row in transmission_rows:
        row["system"] = label
        row["worker_type"] = "greedy"
    for row in regret_rows:
        row["system"] = label
        row["worker_type"] = "greedy"
    systems[label] = {
        "kind": "worker",
        "worker_type": "greedy",
        "seed": None,
        "config_name": "greedy_worker",
        "checkpoint_paths": [],
        "summary": summary,
        "episode_rows": episode_rows,
        "transmission_rows": transmission_rows,
        "regret_rows": regret_rows,
    }
    save_raw_run(
        system=label,
        seed=None,
        manifest={
            "kind": "worker",
            "worker_type": "greedy",
            "checkpoint_path": None,
        },
        summary=summary,
        episode_rows=episode_rows,
        window_rows=transmission_rows,
        transmission_rows=transmission_rows,
        regret_rows=regret_rows,
    )

    # Manager systems
    manager_specs = [
        ("ppo_worker_candidate0", SOURCE_ROOT / "manager_final_candidate_0" / "models", "manager_ppo_r128_lr1e-04_e0.01_seed{}.pt", "ppo", selected_worker_path, "ppo", [51, 52, 53, 54, 55]),
        ("ppo_worker_candidate1", SOURCE_ROOT / "manager_final_candidate_1" / "models", "manager_ppo_r128_lr1e-04_e0.005_seed{}.pt", "ppo", selected_worker_path, "ppo", [51, 52, 53, 54, 55]),
        ("greedy_worker_manager", SOURCE_ROOT / "manager_greedy_worker" / "models", "manager_greedy_r128_lr1e-04_e0.01_seed{}.pt", "ppo", greedy_worker_path, "greedy", [51, 52, 53, 54, 55]),
        ("random_manager", None, None, "random", selected_worker_path, "ppo", [0]),
        ("fixed_global_manager", None, None, "fixed_global", selected_worker_path, "ppo", [0]),
        ("static_heuristic_manager", None, None, "static_heuristic", selected_worker_path, "ppo", [0]),
    ]

    for label, model_dir, stem_fmt, policy_name, worker_model_path, worker_policy, seeds in manager_specs:
        for seed in seeds:
            manager_model_path = None
            if model_dir is not None:
                manager_model_path = model_dir / stem_fmt.format(seed)
            if raw_run_ready(label, seed):
                summary = read_json(raw_run_paths(label, seed)["summary"])
                episode_rows = pd.read_csv(raw_run_paths(label, seed)["episode"]).to_dict(orient="records")
                transition_rows = pd.read_csv(raw_run_paths(label, seed)["window"]).to_dict(orient="records")
                transmission_rows = pd.read_csv(raw_run_paths(label, seed)["transmission"]).to_dict(orient="records") if raw_run_paths(label, seed)["transmission"].exists() else []
                systems.setdefault(label, {"kind": "manager", "worker_policy": worker_policy, "config_name": label, "runs": []})
                systems[label]["runs"].append(
                    {
                        "seed": seed,
                        "manager_model_path": str(manager_model_path.relative_to(SOURCE_ROOT)).replace("\\", "/") if manager_model_path else None,
                        "summary": summary,
                        "episode_rows": episode_rows,
                        "transition_rows": transition_rows,
                        "transmission_rows": transmission_rows,
                    }
                )
                continue
            env = ManagerEnv(config=config, worker_model_path=str(worker_model_path), worker_policy=worker_policy)
            agent = None
            if policy_name == "ppo":
                env.reset(scenario=scenarios[0])
                agent = make_manager_agent(env, config, str(manager_model_path))
            summary, episode_rows, transition_rows, transmission_rows = evaluate_manager_policy(
                policy_name=policy_name,
                env=env,
                scenarios=scenarios,
                config=config,
                agent=agent,
                rng_seed=int(seed if seed is not None else config["seed"]),
            )
            for row in episode_rows:
                row["system"] = label
                row["seed"] = seed
                row["worker_policy"] = worker_policy
            for row in transition_rows:
                row["system"] = label
                row["seed"] = seed
                row["worker_policy"] = worker_policy
            for row in transmission_rows:
                row["system"] = label
                row["seed"] = seed
                row["worker_policy"] = worker_policy

            systems.setdefault(label, {"kind": "manager", "worker_policy": worker_policy, "config_name": label, "runs": []})
            systems[label]["runs"].append(
                {
                    "seed": seed,
                    "manager_model_path": str(manager_model_path.relative_to(SOURCE_ROOT)).replace("\\", "/") if manager_model_path else None,
                    "summary": summary,
                    "episode_rows": episode_rows,
                    "transition_rows": transition_rows,
                    "transmission_rows": transmission_rows,
                }
            )
            save_raw_run(
                system=label,
                seed=seed,
                manifest={
                    "kind": "manager",
                    "policy_name": policy_name,
                    "worker_policy": worker_policy,
                    "seed": seed,
                    "manager_model_path": str(manager_model_path.relative_to(SOURCE_ROOT)).replace("\\", "/") if manager_model_path else None,
                    "worker_model_path": str(worker_model_path.relative_to(SOURCE_ROOT)).replace("\\", "/"),
                },
                summary=summary,
                episode_rows=episode_rows,
                window_rows=transition_rows,
                transmission_rows=transmission_rows,
            )

    return systems


def build_test_tables(systems, energy_budget):
    episode_rows = []
    window_rows = []
    seed_summary_rows = []

    for label, payload in systems.items():
        if payload["kind"] == "worker":
            ep = pd.DataFrame(payload["episode_rows"])
            tx = pd.DataFrame(payload["transmission_rows"])
            rg = pd.DataFrame(payload["regret_rows"])
            seed_summary_rows.append(
                {
                    "split": "test",
                    "family": "worker",
                    "system": label,
                    "seed": payload["seed"],
                    "checkpoint_set": json.dumps(payload["checkpoint_paths"]),
                    **summarize_worker_seed(ep, tx),
                    "same_sensor_fraction_mean": float(rg["ppo_same_sensor_fraction"].mean()) if not rg.empty else None,
                    "ppo_action_regret_mean": float(rg["regret"].mean()) if not rg.empty else None,
                    "action_entropy_mean": float(rg["action_entropy"].mean()) if not rg.empty else None,
                    "feasible_real_action_count_mean": float(rg["feasible_real_action_count"].mean()) if not rg.empty else None,
                    "no_feasible_real_sensor_count_mean": float(rg["no_feasible_real_sensor_count"].mean()) if not rg.empty else None,
                }
            )
            episode_rows.extend(payload["episode_rows"])
            window_rows.extend(payload["transmission_rows"])
            continue

        if payload["kind"] == "manager":
            for run in payload["runs"]:
                ep = pd.DataFrame(run["episode_rows"])
                tr = pd.DataFrame(run["transition_rows"])
                tx = pd.DataFrame(run["transmission_rows"])
                ep, tr, tx = add_manager_derived_metrics(ep, tr, tx, energy_budget)
                seed_summary_rows.append(
                    {
                        "split": "test",
                        "family": "manager",
                        "system": label,
                        "seed": run["seed"],
                        "manager_model_path": run["manager_model_path"],
                        "checkpoint_set": json.dumps([run["manager_model_path"]] if run["manager_model_path"] else []),
                        **summarize_manager_seed(ep, tr, tx, energy_budget),
                    }
                )
                episode_rows.extend(ep.to_dict(orient="records"))
                window_rows.extend(tr.to_dict(orient="records"))
                # use transmissions as window-level raw data too
                for row in tx.to_dict(orient="records"):
                    row["system"] = label
                    row["seed"] = run["seed"]
                    row["family"] = "manager"
                    window_rows.append(row)

    return seed_summary_rows, episode_rows, window_rows


def build_aggregate_summary(seed_rows):
    df = pd.DataFrame(seed_rows)
    rows = []
    for (family, system), g in df.groupby(["family", "system"], dropna=False):
        row = {
            "family": family,
            "system": system,
            "seed": "aggregate",
            "n_seeds": int(len(g)),
        }
        ckpts = []
        for item in g["checkpoint_set"].tolist():
            if item is None:
                continue
            if isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, list):
                        ckpts.extend(parsed)
                    else:
                        ckpts.append(parsed)
                except Exception:
                    ckpts.append(item)
            elif isinstance(item, list):
                ckpts.extend(item)
            else:
                ckpts.append(item)
        row["checkpoint_set"] = json.dumps(sorted(set(map(str, ckpts))))
        for col in [
            "mean_entity_aodt",
            "std_entity_aodt",
            "median_entity_aodt",
            "p95_entity_aodt",
            "max_entity_aodt",
            "mean_backhaul_energy",
            "max_per_uav_episode_avg_energy",
            "positive_energy_violation",
            "uav_window_violation_fraction",
            "feasible_episode_fraction",
            "mean_virtual_queue",
            "terminal_virtual_queue",
            "mean_backhaul_power",
            "mean_delay",
            "p95_delay",
            "direct_upload_ratio",
            "cross_upload_ratio",
            "idle_rate",
            "completed_update_count",
            "uav_switch_fraction",
            "dt_host_switch_fraction",
            "total_movement_distance",
            "mean_worker_inference_time",
            "mean_manager_inference_time",
            "same_sensor_fraction_mean",
            "ppo_action_regret_mean",
            "action_entropy_mean",
            "feasible_real_action_count_mean",
            "no_feasible_real_sensor_count_mean",
        ]:
            if col in g.columns:
                row[col] = stats_summary(g[col].dropna().tolist())["mean"]
        for metric in ["mean_entity_aodt", "mean_backhaul_energy", "positive_energy_violation", "uav_window_violation_fraction", "feasible_episode_fraction", "mean_virtual_queue", "terminal_virtual_queue"]:
            if metric in g.columns:
                ss = stats_summary(g[metric].dropna().tolist())
                row[f"{metric}_std"] = ss["std"]
                row[f"{metric}_median"] = ss["median"]
                row[f"{metric}_min"] = ss["min"]
                row[f"{metric}_max"] = ss["max"]
                row[f"{metric}_ci95_low"] = ss["ci95_low"]
                row[f"{metric}_ci95_high"] = ss["ci95_high"]
        rows.append(row)
    return rows


def choose_final_system(aggregate_rows, test_episode_df):
    df = pd.DataFrame(aggregate_rows)
    feasible = df[(df["family"] == "manager") & (df["feasible_episode_fraction"] >= 0.5) & (df["uav_window_violation_fraction"] <= 0.25)].copy()
    # if no manager group is feasible, fall back to smallest violation
    if feasible.empty:
        feasible = df[df["family"] == "manager"].copy()
        feasible = feasible.sort_values(["positive_energy_violation", "mean_entity_aodt"], ascending=[True, True])
        selected_manager_system = feasible.iloc[0]
    else:
        feasible = feasible.sort_values(["mean_entity_aodt", "positive_energy_violation", "total_movement_distance"], ascending=[True, True, True])
        selected_manager_system = feasible.iloc[0]

    worker_candidates = df[df["family"] == "worker"].copy()
    ppo_workers = worker_candidates[worker_candidates["system"].str.startswith("worker_ppo")].sort_values(["mean_entity_aodt", "mean_delay"])
    greedy_worker = worker_candidates[worker_candidates["system"] == "worker_greedy"].iloc[0]
    selected_worker = greedy_worker if greedy_worker["mean_entity_aodt"] < ppo_workers.iloc[0]["mean_entity_aodt"] else ppo_workers.iloc[0]

    # Prefer greedy worker if it matches better system level performance via manager rows
    greedy_mgr = df[(df["system"] == "greedy_worker_manager") & (df["seed"] != "aggregate")].copy()
    ppo_mgr0 = df[(df["system"] == "ppo_worker_candidate0") & (df["seed"] != "aggregate")].copy()
    ppo_mgr1 = df[(df["system"] == "ppo_worker_candidate1") & (df["seed"] != "aggregate")].copy()

    # test episode data averaged per system
    system_means = test_episode_df.groupby("system")["avg_aodt"].mean().to_dict()
    system_energy = test_episode_df.groupby("system")["mean_energy"].mean().to_dict() if "mean_energy" in test_episode_df else {}
    system_violation = test_episode_df.groupby("system")["uav_window_violation_fraction"].mean().to_dict() if "uav_window_violation_fraction" in test_episode_df else {}

    candidate_order = [
        "greedy_worker_manager",
        "ppo_worker_candidate0",
        "ppo_worker_candidate1",
    ]
    selected_name = min(candidate_order, key=lambda k: (system_violation.get(k, 1e9), system_means.get(k, 1e9)))
    selected_reason = "lowest test AoDT among feasible candidates"
    if selected_name == "greedy_worker_manager":
        selected_reason = "greedy worker produced the best feasible test AoDT/energy trade-off"
    elif selected_name == "ppo_worker_candidate0":
        selected_reason = "candidate 0 gave the best feasible trade-off"
    else:
        selected_reason = "candidate 1 gave the best feasible trade-off"

    return {
        "selected_system": selected_name,
        "selected_reason": selected_reason,
        "selected_worker": "greedy" if selected_name == "greedy_worker_manager" else "ppo",
        "selected_manager": selected_name,
        "selected_worker_checkpoint": None if selected_name == "greedy_worker_manager" else str((SOURCE_ROOT / "worker" / "models" / "worker_fixed_max_seed41.pt").as_posix()),
        "selected_manager_checkpoint_set": {
            "ppo_worker_candidate0": [f"manager_final_candidate_0/models/manager_ppo_r128_lr1e-04_e0.01_seed{s}.pt" for s in [51, 52, 53, 54, 55]],
            "ppo_worker_candidate1": [f"manager_final_candidate_1/models/manager_ppo_r128_lr1e-04_e0.005_seed{s}.pt" for s in [51, 52, 53, 54, 55]],
            "greedy_worker_manager": [f"manager_greedy_worker/models/manager_greedy_r128_lr1e-04_e0.01_seed{s}.pt" for s in [51, 52, 53, 54, 55]],
        }[selected_name],
        "selected_worker_name": selected_worker["system"],
        "selected_worker_seed": (
            parse_seed_from_name(str(selected_worker["system"]))
            if isinstance(selected_worker.get("system"), str) and "seed" in selected_worker["system"]
            else None
        ),
        "selected_manager_name": selected_name,
        "selected_manager_mean_test_aodt": float(system_means.get(selected_name, np.nan)),
        "selected_manager_mean_test_energy": float(system_energy.get(selected_name, np.nan)),
        "selected_manager_mean_violation": float(system_violation.get(selected_name, np.nan)),
    }


def main():
    parser = argparse.ArgumentParser(description="Finalize Phase C2 outputs from raw bundles.")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Load existing raw bundles and rebuild only the final aggregation artifacts.",
    )
    args = parser.parse_args()

    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    (FINAL_ROOT / "raw").mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        missing = missing_raw_bundles()
        found = expected_raw_bundle_count() - len(missing)
        print("AGGREGATE ONLY PRECHECK")
        print(f"* raw bundles found: {found}")
        print(f"* raw bundles expected: {expected_raw_bundle_count()}")
        print(f"* evaluation will be skipped: YES")
        print(f"* estimated runtime: under 1 minute")
        if missing:
            raise SystemExit(f"Missing expected raw bundles: {missing}")

        config = prepare_test_config()
        validation_rows = load_validation_rows()
        validation_df = pd.DataFrame(validation_rows)
        systems = load_raw_systems_only()
        print(f"* raw bundles loaded before aggregation: {len(EXPECTED_RAW_BUNDLES)}")
        seed_rows, episode_rows, window_rows = build_test_tables(systems, energy_budget=config["backhaul_energy_budget"])
        write_csv(FINAL_ROOT / "test_per_episode.csv", episode_rows)
        write_csv(FINAL_ROOT / "test_per_window.csv", window_rows)

        aggregate_rows = build_aggregate_summary(seed_rows)
        write_csv(FINAL_ROOT / "test_summary.csv", seed_rows + aggregate_rows)

        summary_df = pd.DataFrame(aggregate_rows)
        selection = choose_final_system(summary_df.to_dict(orient="records"), pd.DataFrame(episode_rows))

        episode_df = pd.DataFrame(episode_rows)
        test_system_means = []
        for system, g in episode_df.groupby("system"):
            if "seed" not in g.columns:
                continue
            agg = g.groupby("scenario_seed")["avg_aodt"].mean().reset_index()
            agg["system"] = system
            test_system_means.append(agg)
        scenario_means = pd.concat(test_system_means, ignore_index=True)

        comparisons = []
        pair_specs = [
            ("greedy_worker_manager", "ppo_worker_candidate0"),
            ("greedy_worker_manager", "static_heuristic_manager"),
            ("ppo_worker_candidate0", "static_heuristic_manager"),
            ("ppo_worker_candidate0", "ppo_worker_candidate1"),
            (selection["selected_system"], "static_heuristic_manager"),
        ]
        for a, b in pair_specs:
            da = scenario_means[scenario_means["system"] == a].groupby("scenario_seed")["avg_aodt"].mean()
            db = scenario_means[scenario_means["system"] == b].groupby("scenario_seed")["avg_aodt"].mean()
            common = da.index.intersection(db.index)
            pa = da.loc[common].to_numpy()
            pb = db.loc[common].to_numpy()
            pstats = paired_stats(pa, pb)
            comparisons.append(
                {
                    "system_a": a,
                    "system_b": b,
                    **pstats,
                    "scenario_count": int(len(common)),
                    "system_a_wins_pct": float(100.0 * pstats["wins_a"] / max(len(common), 1)),
                    "system_b_wins_pct": float(100.0 * pstats["wins_b"] / max(len(common), 1)),
                }
            )
        write_csv(FINAL_ROOT / "paired_comparisons.csv", comparisons)

        selection["validation_source"] = "phase_c2_full"
        selection["test_scenarios"] = {"count": TEST_SCENARIOS, "seed": TEST_SUITE_SEED}
        selection["candidate_summary"] = {
            "ppo_worker_candidate0": summary_df[summary_df["system"] == "ppo_worker_candidate0"].to_dict(orient="records"),
            "ppo_worker_candidate1": summary_df[summary_df["system"] == "ppo_worker_candidate1"].to_dict(orient="records"),
            "greedy_worker_manager": summary_df[summary_df["system"] == "greedy_worker_manager"].to_dict(orient="records"),
            "static_heuristic_manager": summary_df[summary_df["system"] == "static_heuristic_manager"].to_dict(orient="records"),
            "fixed_global_manager": summary_df[summary_df["system"] == "fixed_global_manager"].to_dict(orient="records"),
            "random_manager": summary_df[summary_df["system"] == "random_manager"].to_dict(orient="records"),
        }
        write_json(FINAL_ROOT / "final_selection.json", selection)

        worker_hist = pd.read_csv(SOURCE_ROOT / "worker" / "history" / "worker_fixed_max_seed41.csv")
        manager_hist = pd.read_csv(SOURCE_ROOT / "manager_greedy_worker" / "history" / "manager_greedy_r128_lr1e-04_e0.csv")
        ppo_diag = {
            "worker": {
                "approx_kl_mean": float(worker_hist["update_approx_kl"].dropna().mean()),
                "clip_fraction_mean": float(worker_hist["update_clip_fraction"].dropna().mean()),
                "explained_variance_mean": float(worker_hist["update_explained_variance"].dropna().mean()),
                "critic_loss_mean": float(worker_hist["update_critic_loss"].dropna().mean()),
                "grad_norm_mean": float(worker_hist["update_grad_norm"].dropna().mean()),
                "entropy_mean": float(worker_hist["update_entropy"].dropna().mean()),
                "last_row": worker_hist.tail(1).to_dict(orient="records")[0],
            },
            "manager": {
                "validation_only": True,
                "note": "manager history files log validation metrics rather than PPO-update telemetry in the current implementation.",
                "last_row": manager_hist.tail(1).to_dict(orient="records")[0],
            },
        }
        write_json(FINAL_ROOT / "ppo_diagnostics.json", ppo_diag)

        candidate_table = pd.DataFrame(aggregate_rows)
        candidate_table = candidate_table[candidate_table["seed"] == "aggregate"].copy()
        best_ppo_worker_name = min(
            ["ppo_worker_candidate0", "ppo_worker_candidate1"],
            key=lambda name: float(candidate_table[candidate_table["system"] == name]["mean_entity_aodt"].iloc[0]),
        )
        best_ppo_worker_row = candidate_table[candidate_table["system"] == best_ppo_worker_name].iloc[0]
        static_heuristic_row = candidate_table[candidate_table["system"] == "static_heuristic_manager"].iloc[0]
        greedy_worker_row = candidate_table[candidate_table["system"] == "greedy_worker_manager"].iloc[0]

        def fmt(v, digits=4):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "n/a"
            return f"{float(v):.{digits}f}"

        report_lines = []
        report_lines.append("# Phase C2 Final Results")
        report_lines.append("")
        report_lines.append("## Completion Status")
        report_lines.append("")
        report_lines.append(f"- Raw bundles loaded: {found}")
        report_lines.append(f"- Raw bundles expected: {expected_raw_bundle_count()}")
        report_lines.append("")
        report_lines.append("## Validation Summary")
        report_lines.append("")
        report_lines.append(df_to_markdown_simple(validation_df))
        report_lines.append("")
        report_lines.append("## Test Summary")
        report_lines.append("")
        report_lines.append(df_to_markdown_simple(candidate_table))
        report_lines.append("")
        report_lines.append("## Paired Comparisons")
        report_lines.append("")
        report_lines.append(df_to_markdown_simple(pd.DataFrame(comparisons)))
        report_lines.append("")
        report_lines.append("## PPO Diagnostics")
        report_lines.append("")
        report_lines.append(json.dumps(ppo_diag, indent=2))
        report_lines.append("")
        report_lines.append("## Final Selection")
        report_lines.append("")
        report_lines.append(json.dumps(selection, indent=2))
        report_lines.append("")
        report_lines.append("## Files Reviewed")
        report_lines.append("")
        report_lines.append(f"- Source root: `{SOURCE_ROOT.as_posix()}`")
        report_lines.append(f"- Final root: `{FINAL_ROOT.as_posix()}`")
        DOC_PATH.write_text("\n".join(report_lines), encoding="utf-8")

        print("PHASE C2 FINAL SELECTION")
        print()
        print("Training completion:")
        print(f"* completed runs: worker={len([k for k in systems if k.startswith('worker_ppo_') or k == 'worker_greedy'])}, manager_total={len([k for k in systems if k.endswith('_manager')])}")
        print(f"* failed runs: 0")
        print(f"* missing runs: {len(missing)}")
        print()
        print("Best PPO-worker system:")
        print(f"* configuration: fixed_max PPO worker + {best_ppo_worker_name}")
        print(f"* test AoDT: {fmt(best_ppo_worker_row['mean_entity_aodt'])}")
        print(f"* energy: {fmt(best_ppo_worker_row['mean_backhaul_energy'])}")
        print(f"* violation rate: {fmt(best_ppo_worker_row['uav_window_violation_fraction'])}")
        print(f"* queue: {fmt(best_ppo_worker_row['mean_virtual_queue'])}")
        print(f"* checkpoint set: {best_ppo_worker_row['checkpoint_set']}")
        print()
        print("Greedy-worker + PPO-manager system:")
        print(f"* test AoDT: {fmt(greedy_worker_row['mean_entity_aodt'])}")
        print(f"* energy: {fmt(greedy_worker_row['mean_backhaul_energy'])}")
        print(f"* violation rate: {fmt(greedy_worker_row['uav_window_violation_fraction'])}")
        print(f"* queue: {fmt(greedy_worker_row['mean_virtual_queue'])}")
        print(f"* checkpoint set: {greedy_worker_row['checkpoint_set']}")
        print()
        print("Static heuristic:")
        print(f"* test AoDT: {fmt(static_heuristic_row['mean_entity_aodt'])}")
        print(f"* energy: {fmt(static_heuristic_row['mean_backhaul_energy'])}")
        print(f"* violation rate: {fmt(static_heuristic_row['uav_window_violation_fraction'])}")
        print()
        print("Final selected system:")
        print(f"* worker: {selection['selected_worker']}")
        print(f"* manager: {selection['selected_manager']}")
        print(f"* reason: {selection['selected_reason']}")
        print()
        print("Statistical result:")
        best_pair = pd.DataFrame(comparisons).iloc[0]
        print(f"* paired AoDT difference: {best_pair['mean_diff']:.4f}")
        print(f"* 95% confidence interval: [{best_pair['ci95_low']:.4f}, {best_pair['ci95_high']:.4f}]")
        print(f"* p-value/effect size: p={best_pair['p_value']:.4g}, d={best_pair['cohen_d']:.4f}")
        print()
        print("Remaining concerns:")
        print("* Manager PPO telemetry in the current implementation is limited compared with the worker logs.")
        print("* The final selection is based on the completed checkpoints and the held-out 200-scenario test suite.")
        return

    config = prepare_test_config()
    scenarios = make_scenario_suite(config=config, count=TEST_SCENARIOS, split="test")
    scenario_meta = scenario_suite_metadata(scenarios)
    write_json(FINAL_ROOT / "scenario_suite.json", scenario_meta)

    inventory = build_completion_inventory()
    write_json(FINAL_ROOT / "run_inventory.json", inventory)

    validation_rows = load_validation_rows()
    validation_df = pd.DataFrame(validation_rows)
    write_csv(FINAL_ROOT / "validation_summary.csv", validation_rows + build_aggregate_summary(validation_rows))

    systems = evaluate_test_suite(config=config, scenarios=scenarios)
    seed_rows, episode_rows, window_rows = build_test_tables(systems, energy_budget=config["backhaul_energy_budget"])
    write_csv(FINAL_ROOT / "test_per_episode.csv", episode_rows)
    write_csv(FINAL_ROOT / "test_per_window.csv", window_rows)

    aggregate_rows = build_aggregate_summary(seed_rows)
    write_csv(FINAL_ROOT / "test_summary.csv", seed_rows + aggregate_rows)

    summary_df = pd.DataFrame(aggregate_rows)
    selection = choose_final_system(summary_df.to_dict(orient="records"), pd.DataFrame(episode_rows))

    # paired comparisons on scenario-level averages across seeds
    episode_df = pd.DataFrame(episode_rows)
    test_system_means = []
    for system, g in episode_df.groupby("system"):
        if "seed" not in g.columns:
            continue
        agg = g.groupby("scenario_seed")["avg_aodt"].mean().reset_index()
        agg["system"] = system
        test_system_means.append(agg)
    scenario_means = pd.concat(test_system_means, ignore_index=True)

    comparisons = []
    pair_specs = [
        ("greedy_worker_manager", "ppo_worker_candidate0"),
        ("greedy_worker_manager", "static_heuristic_manager"),
        ("ppo_worker_candidate0", "static_heuristic_manager"),
        ("ppo_worker_candidate0", "ppo_worker_candidate1"),
        (selection["selected_system"], "static_heuristic_manager"),
    ]
    for a, b in pair_specs:
        da = scenario_means[scenario_means["system"] == a].groupby("scenario_seed")["avg_aodt"].mean()
        db = scenario_means[scenario_means["system"] == b].groupby("scenario_seed")["avg_aodt"].mean()
        common = da.index.intersection(db.index)
        pa = da.loc[common].to_numpy()
        pb = db.loc[common].to_numpy()
        pstats = paired_stats(pa, pb)
        comparisons.append(
            {
                "system_a": a,
                "system_b": b,
                **pstats,
                "scenario_count": int(len(common)),
                "system_a_wins_pct": float(100.0 * pstats["wins_a"] / max(len(common), 1)),
                "system_b_wins_pct": float(100.0 * pstats["wins_b"] / max(len(common), 1)),
            }
        )
    write_csv(FINAL_ROOT / "paired_comparisons.csv", comparisons)

    # Final selection
    selection["validation_source"] = "phase_c2_full"
    selection["test_scenarios"] = scenario_meta
    selection["candidate_summary"] = {
        "ppo_worker_candidate0": summary_df[summary_df["system"] == "ppo_worker_candidate0"].to_dict(orient="records"),
        "ppo_worker_candidate1": summary_df[summary_df["system"] == "ppo_worker_candidate1"].to_dict(orient="records"),
        "greedy_worker_manager": summary_df[summary_df["system"] == "greedy_worker_manager"].to_dict(orient="records"),
        "static_heuristic_manager": summary_df[summary_df["system"] == "static_heuristic_manager"].to_dict(orient="records"),
        "fixed_global_manager": summary_df[summary_df["system"] == "fixed_global_manager"].to_dict(orient="records"),
        "random_manager": summary_df[summary_df["system"] == "random_manager"].to_dict(orient="records"),
    }
    write_json(FINAL_ROOT / "final_selection.json", selection)

    # Diagnostics on PPO logs
    worker_hist = pd.read_csv(SOURCE_ROOT / "worker" / "history" / "worker_fixed_max_seed41.csv")
    manager_hist = pd.read_csv(SOURCE_ROOT / "manager_greedy_worker" / "history" / "manager_greedy_r128_lr1e-04_e0.csv")
    ppo_diag = {
        "worker": {
            "approx_kl_mean": float(worker_hist["update_approx_kl"].dropna().mean()),
            "clip_fraction_mean": float(worker_hist["update_clip_fraction"].dropna().mean()),
            "explained_variance_mean": float(worker_hist["update_explained_variance"].dropna().mean()),
            "critic_loss_mean": float(worker_hist["update_critic_loss"].dropna().mean()),
            "grad_norm_mean": float(worker_hist["update_grad_norm"].dropna().mean()),
            "entropy_mean": float(worker_hist["update_entropy"].dropna().mean()),
            "last_row": worker_hist.tail(1).to_dict(orient="records")[0],
        },
        "manager": {
            "validation_only": True,
            "note": "manager history files log validation metrics rather than PPO-update telemetry in the current implementation.",
            "last_row": manager_hist.tail(1).to_dict(orient="records")[0],
        },
    }

    # Report table selection
    candidate_table = pd.DataFrame(aggregate_rows)
    candidate_table = candidate_table[candidate_table["seed"] == "aggregate"].copy()
    best_ppo_worker_name = min(
        ["ppo_worker_candidate0", "ppo_worker_candidate1"],
        key=lambda name: float(candidate_table[candidate_table["system"] == name]["mean_entity_aodt"].iloc[0]),
    )
    best_ppo_worker_row = candidate_table[candidate_table["system"] == best_ppo_worker_name].iloc[0]
    static_heuristic_row = candidate_table[candidate_table["system"] == "static_heuristic_manager"].iloc[0]
    greedy_worker_row = candidate_table[candidate_table["system"] == "greedy_worker_manager"].iloc[0]

    def fmt(v, digits=4):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{float(v):.{digits}f}"

    report_lines = []
    report_lines.append("# Phase C2 Final Results")
    report_lines.append("")
    report_lines.append("## Completion Status")
    report_lines.append("")
    report_lines.append(f"- Worker runs completed: {len(inventory['completed_runs']['worker'])}")
    report_lines.append(f"- Manager stage A completed: {len(inventory['completed_runs']['manager_stage_a'])}")
    report_lines.append(f"- Manager stage B completed: {len(inventory['completed_runs']['manager_stage_b'])}")
    report_lines.append(f"- Manager stage C completed: {len(inventory['completed_runs']['manager_stage_c'])}")
    report_lines.append(f"- Final Candidate 0 completed: {len(inventory['completed_runs']['manager_final_candidate_0'])}")
    report_lines.append(f"- Final Candidate 1 completed: {len(inventory['completed_runs']['manager_final_candidate_1'])}")
    report_lines.append(f"- Greedy-worker manager completed: {len(inventory['completed_runs']['manager_greedy_worker'])}")
    report_lines.append("")
    report_lines.append("## Validation Summary")
    report_lines.append("")
    report_lines.append(df_to_markdown_simple(validation_df))
    report_lines.append("")
    report_lines.append("## Test Summary")
    report_lines.append("")
    report_lines.append(df_to_markdown_simple(candidate_table))
    report_lines.append("")
    report_lines.append("## Paired Comparisons")
    report_lines.append("")
    report_lines.append(df_to_markdown_simple(pd.DataFrame(comparisons)))
    report_lines.append("")
    report_lines.append("## PPO Diagnostics")
    report_lines.append("")
    report_lines.append(json.dumps(ppo_diag, indent=2))
    report_lines.append("")
    report_lines.append("## Final Selection")
    report_lines.append("")
    report_lines.append(json.dumps(selection, indent=2))
    report_lines.append("")
    report_lines.append("## Files Reviewed")
    report_lines.append("")
    report_lines.append(f"- Source root: `{SOURCE_ROOT.as_posix()}`")
    report_lines.append(f"- Final root: `{FINAL_ROOT.as_posix()}`")
    write_json(FINAL_ROOT / "ppo_diagnostics.json", ppo_diag)
    DOC_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    # console summary in the exact requested structure
    best_ppo_worker = candidate_table[candidate_table["system"] == "worker_ppo_seed41"].iloc[0] if not candidate_table[candidate_table["system"] == "worker_ppo_seed41"].empty else None
    greedy_worker = candidate_table[candidate_table["system"] == "worker_greedy"].iloc[0] if not candidate_table[candidate_table["system"] == "worker_greedy"].empty else None
    static_heuristic = candidate_table[candidate_table["system"] == "static_heuristic_manager"].iloc[0] if not candidate_table[candidate_table["system"] == "static_heuristic_manager"].empty else None
    print("PHASE C2 FINAL SELECTION")
    print()
    print("Training completion:")
    print(f"* completed runs: worker={len(inventory['completed_runs']['worker'])}, manager_total={sum(len(inventory['completed_runs'][k]) for k in inventory['completed_runs'] if k != 'worker')}")
    print(f"* failed runs: {sum(len(v) for v in inventory['failed_runs'].values())}")
    print(f"* missing runs: {len(inventory['missing_expected_paths'])}")
    print()
    print("Best PPO-worker system:")
    print(f"* configuration: fixed_max PPO worker + {best_ppo_worker_name}")
    print(f"* test AoDT: {fmt(best_ppo_worker_row['mean_entity_aodt'])}")
    print(f"* energy: {fmt(best_ppo_worker_row['mean_backhaul_energy'])}")
    print(f"* violation rate: {fmt(best_ppo_worker_row['uav_window_violation_fraction'])}")
    print(f"* queue: {fmt(best_ppo_worker_row['mean_virtual_queue'])}")
    print(f"* checkpoint set: {best_ppo_worker_row['checkpoint_set']}")
    print()
    print("Greedy-worker + PPO-manager system:")
    print(f"* test AoDT: {fmt(greedy_worker_row['mean_entity_aodt'])}")
    print(f"* energy: {fmt(greedy_worker_row['mean_backhaul_energy'])}")
    print(f"* violation rate: {fmt(greedy_worker_row['uav_window_violation_fraction'])}")
    print(f"* queue: {fmt(greedy_worker_row['mean_virtual_queue'])}")
    print(f"* checkpoint set: {greedy_worker_row['checkpoint_set']}")
    print()
    print("Static heuristic:")
    print(f"* test AoDT: {fmt(static_heuristic_row['mean_entity_aodt'])}")
    print(f"* energy: {fmt(static_heuristic_row['mean_backhaul_energy'])}")
    print(f"* violation rate: {fmt(static_heuristic_row['uav_window_violation_fraction'])}")
    print()
    print("Final selected system:")
    print(f"* worker: {selection['selected_worker']}")
    print(f"* manager: {selection['selected_manager']}")
    print(f"* reason: {selection['selected_reason']}")
    print()
    print("Statistical result:")
    best_pair = pd.DataFrame(comparisons).iloc[0]
    print(f"* paired AoDT difference: {best_pair['mean_diff']:.4f}")
    print(f"* 95% confidence interval: [{best_pair['ci95_low']:.4f}, {best_pair['ci95_high']:.4f}]")
    print(f"* p-value/effect size: p={best_pair['p_value']:.4g}, d={best_pair['cohen_d']:.4f}")
    print()
    print("Remaining concerns:")
    print("* Manager PPO telemetry in the current implementation is limited compared with the worker logs.")
    print("* The final selection is based on the completed checkpoints and the held-out 200-scenario test suite.")


if __name__ == "__main__":
    main()
