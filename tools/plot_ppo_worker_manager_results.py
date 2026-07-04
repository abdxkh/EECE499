"""Create presentation plots for the PPO-worker + PPO-manager system.

This script reads existing Phase C2 final raw bundles only. It does not
train, evaluate, or modify checkpoints/raw results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "results" / "phase_c2_final"
FULL_DIR = ROOT / "outputs" / "results" / "phase_c2_full"
OUT_DIR = ROOT / "outputs" / "results" / "ppo_worker_manager_presentation"
DOC_PATH = ROOT / "docs" / "PPO_WORKER_MANAGER_PRESENTATION_RESULTS.md"

SYSTEMS = {
    "ppo_worker_candidate0": {
        "label": "PPO worker + PPO manager C0",
        "history_dir": FULL_DIR / "manager_final_candidate_0" / "history",
        "glob": "*.csv",
    },
    "ppo_worker_candidate1": {
        "label": "PPO worker + PPO manager C1",
        "history_dir": FULL_DIR / "manager_final_candidate_1" / "history",
        "glob": "*.csv",
    },
}


def _read_raw_episode_rows(system: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    raw_root = FINAL_DIR / "raw" / system
    for seed_dir in sorted(raw_root.glob("seed*")):
        csv_path = seed_dir / "per_episode.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        frames.append(pd.read_csv(csv_path))
    if not frames:
        raise RuntimeError(f"No raw episode bundles found for {system}")
    return pd.concat(frames, ignore_index=True)


def _read_raw_summaries() -> pd.DataFrame:
    summary_path = FINAL_DIR / "test_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    df = pd.read_csv(summary_path)
    return df[(df["split"] == "test") & (df["system"].isin(SYSTEMS))].copy()


def _read_validation_curves(system: str) -> pd.DataFrame:
    meta = SYSTEMS[system]
    frames: list[pd.DataFrame] = []
    for path in sorted(meta["history_dir"].glob(meta["glob"])):
        frame = pd.read_csv(path)
        frame["system"] = system
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No validation curves found for {system}")
    return pd.concat(frames, ignore_index=True)


def _savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=220)
    plt.close()


def make_plots(summary: pd.DataFrame, episodes: pd.DataFrame, validation: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_summary = summary.copy()
    plot_summary["system_label"] = plot_summary["system"].map(lambda s: SYSTEMS[s]["label"])

    plt.figure(figsize=(8.5, 4.8))
    for system, meta in SYSTEMS.items():
        curve = validation[validation["system"] == system]
        grouped = curve.groupby("episode")["avg_aodt"].agg(["mean", "std"]).reset_index()
        plt.plot(grouped["episode"], grouped["mean"], label=meta["label"])
        plt.fill_between(
            grouped["episode"],
            grouped["mean"] - grouped["std"].fillna(0.0),
            grouped["mean"] + grouped["std"].fillna(0.0),
            alpha=0.16,
        )
    plt.xlabel("Manager training episode")
    plt.ylabel("Validation AoDT")
    plt.title("PPO manager validation AoDT during training")
    plt.grid(alpha=0.25)
    plt.legend()
    _savefig("ppo_manager_validation_aodt.png")

    plt.figure(figsize=(8.5, 4.8))
    for system, meta in SYSTEMS.items():
        curve = validation[validation["system"] == system]
        grouped = curve.groupby("episode")["avg_energy"].agg(["mean", "std"]).reset_index()
        plt.plot(grouped["episode"], grouped["mean"], label=meta["label"])
        plt.fill_between(
            grouped["episode"],
            grouped["mean"] - grouped["std"].fillna(0.0),
            grouped["mean"] + grouped["std"].fillna(0.0),
            alpha=0.16,
        )
    plt.axhline(0.25, color="black", linestyle="--", linewidth=1.0, label="Budget")
    plt.xlabel("Manager training episode")
    plt.ylabel("Validation backhaul energy")
    plt.title("PPO manager validation energy during training")
    plt.grid(alpha=0.25)
    plt.legend()
    _savefig("ppo_manager_validation_energy.png")

    c1 = validation[validation["system"] == "ppo_worker_candidate1"].sort_values("episode").copy()
    c1["running_best_aodt"] = c1["avg_aodt"].cummin()
    best_idx = c1["avg_aodt"].idxmin()
    best = c1.loc[best_idx]
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(c1["episode"], c1["avg_aodt"], marker="o", linewidth=1.8, label="Validation AoDT")
    plt.plot(c1["episode"], c1["running_best_aodt"], linewidth=2.4, label="Best AoDT so far")
    plt.scatter([best["episode"]], [best["avg_aodt"]], color="#b23b3b", zorder=3, label="Best checkpoint")
    plt.annotate(
        f"best {best['avg_aodt']:.3f}\nepisode {int(best['episode'])}",
        (best["episode"], best["avg_aodt"]),
        xytext=(12, 16),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    plt.xlabel("Manager training episode")
    plt.ylabel("Validation AoDT")
    plt.title("PPO worker + PPO manager C1: AoDT improvement over training")
    plt.grid(alpha=0.25)
    plt.legend()
    _savefig("c1_aodt_improvement_over_time.png")

    c1["running_best_energy"] = c1["avg_energy"].cummin()
    best_energy_idx = c1["avg_energy"].idxmin()
    best_energy = c1.loc[best_energy_idx]
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(c1["episode"], c1["avg_energy"], marker="o", linewidth=1.8, label="Validation backhaul energy")
    plt.plot(c1["episode"], c1["running_best_energy"], linewidth=2.4, label="Best energy so far")
    plt.axhline(0.25, color="black", linestyle="--", linewidth=1.0, label="Energy budget")
    plt.scatter(
        [best_energy["episode"]],
        [best_energy["avg_energy"]],
        color="#b23b3b",
        zorder=3,
        label="Lowest energy point",
    )
    plt.annotate(
        f"lowest {best_energy['avg_energy']:.3f}\nepisode {int(best_energy['episode'])}",
        (best_energy["episode"], best_energy["avg_energy"]),
        xytext=(12, 16),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    plt.xlabel("Manager training episode")
    plt.ylabel("Validation backhaul energy")
    plt.title("PPO worker + PPO manager C1: backhaul energy over training")
    plt.grid(alpha=0.25)
    plt.legend()
    _savefig("c1_backhaul_energy_over_time.png")


def write_report(summary: pd.DataFrame, episodes: pd.DataFrame, validation: pd.DataFrame) -> None:
    aggregate_rows = []
    for system, meta in SYSTEMS.items():
        seed_rows = summary[summary["system"] == system]
        episode_rows = episodes[episodes["system"] == system]
        aggregate_rows.append(
            {
                "system": meta["label"],
                "seeds": len(seed_rows),
                "test_mean_aodt": seed_rows["mean_entity_aodt"].mean(),
                "test_std_aodt_across_seeds": seed_rows["mean_entity_aodt"].std(ddof=1),
                "test_median_episode_aodt": episode_rows["avg_aodt"].median(),
                "test_p95_episode_aodt": episode_rows["avg_aodt"].quantile(0.95),
                "test_mean_energy": seed_rows["mean_backhaul_energy"].mean(),
                "test_positive_violation": seed_rows["positive_energy_violation"].mean(),
                "test_violation_fraction": seed_rows["uav_window_violation_fraction"].mean(),
                "test_mean_delay": seed_rows["mean_delay"].mean(),
                "test_p95_delay": seed_rows["p95_delay"].mean(),
                "direct_upload_ratio": seed_rows["direct_upload_ratio"].mean(),
                "cross_upload_ratio": seed_rows["cross_upload_ratio"].mean(),
                "idle_rate": seed_rows["idle_rate"].mean(),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(OUT_DIR / "ppo_worker_manager_summary.csv", index=False)

    report_columns = [
        "system",
        "seeds",
        "test_mean_aodt",
        "test_std_aodt_across_seeds",
        "test_p95_episode_aodt",
        "test_mean_energy",
        "test_violation_fraction",
        "test_p95_delay",
    ]
    table = aggregate[report_columns].copy()
    for column in table.columns:
        if column != "system":
            table[column] = table[column].map(lambda value: f"{value:.4f}" if isinstance(value, float) else str(value))
    header = "| " + " | ".join(report_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(report_columns)) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in report_columns) + " |"
        for _, row in table.iterrows()
    ]
    markdown_table = "\n".join([header, separator, *body])

    best_row = summary.loc[summary["mean_entity_aodt"].idxmin()]
    paired_path = FINAL_DIR / "paired_comparisons.csv"
    paired_text = ""
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        pair = paired[
            ((paired["system_a"] == "ppo_worker_candidate0") & (paired["system_b"] == "ppo_worker_candidate1"))
            | ((paired["system_a"] == "ppo_worker_candidate1") & (paired["system_b"] == "ppo_worker_candidate0"))
        ]
        if not pair.empty:
            p = pair.iloc[0]
            paired_text = (
                f"- Candidate comparison: `{p.system_a}` minus `{p.system_b}` mean paired AoDT "
                f"difference = `{p.mean_diff:.4f}`, 95% CI `[{p.ci95_low:.4f}, {p.ci95_high:.4f}]`, "
                f"p-value `{p.p_value:.4g}`.\n"
            )

    lines = [
        "# PPO Worker + PPO Manager Presentation Results",
        "",
        "These plots are generated from saved Phase C2 final artifacts only. No training or evaluation was rerun.",
        "",
        "## Systems Plotted",
        "",
        "- `ppo_worker_candidate0`: PPO worker with PPO manager candidate 0, entropy coefficient 0.01.",
        "- `ppo_worker_candidate1`: PPO worker with PPO manager candidate 1, entropy coefficient 0.005.",
        "",
        "## Test Summary",
        "",
        markdown_table,
        "",
        "## Best PPO-Only Seed",
        "",
        (
            f"- Best seed by test AoDT: `{best_row.system}`, seed `{int(best_row.seed)}` "
            f"with mean AoDT `{best_row.mean_entity_aodt:.4f}`, energy `{best_row.mean_backhaul_energy:.4f}`, "
            f"and violation fraction `{best_row.uav_window_violation_fraction:.4f}`."
        ),
        paired_text.rstrip(),
        "",
        "## Generated Figures",
        "",
        "- `ppo_manager_validation_aodt.png`: validation AoDT during PPO-manager training.",
        "- `ppo_manager_validation_energy.png`: validation energy during PPO-manager training.",
        "- `c1_aodt_improvement_over_time.png`: C1 validation AoDT and running-best AoDT over training.",
        "- `c1_backhaul_energy_over_time.png`: C1 validation backhaul energy and energy budget over training.",
        "",
        "## Presentation Note",
        "",
        (
            "This is the PPO-worker + PPO-manager result set. It is useful for explaining how PPO was implemented "
            "and evaluated, but it should be presented as the PPO-only hierarchical baseline rather than as the "
            "final selected controller because the final selected system uses the greedy worker with the PPO manager."
        ),
    ]
    DOC_PATH.write_text("\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8")


def main() -> None:
    summary = _read_raw_summaries()
    episode_frames = [_read_raw_episode_rows(system) for system in SYSTEMS]
    episodes = pd.concat(episode_frames, ignore_index=True)
    validation = pd.concat([_read_validation_curves(system) for system in SYSTEMS], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "ppo_worker_manager_seed_summary.csv", index=False)
    episodes.to_csv(OUT_DIR / "ppo_worker_manager_per_episode.csv", index=False)
    validation.to_csv(OUT_DIR / "ppo_worker_manager_validation_history.csv", index=False)
    make_plots(summary, episodes, validation)
    write_report(summary, episodes, validation)

    print("PPO worker + PPO manager presentation artifacts created")
    print(f"Output directory: {OUT_DIR}")
    print(f"Report: {DOC_PATH}")


if __name__ == "__main__":
    main()
