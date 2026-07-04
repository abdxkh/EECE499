import numpy as np


def summarize_delays(delays, slot_duration):
    delays = np.asarray(delays, dtype=np.float32)
    if len(delays) == 0:
        return {
            "count": 0,
            "mean_delay": 0.0,
            "median_delay": 0.0,
            "p90_delay": 0.0,
            "p95_delay": 0.0,
            "p99_delay": 0.0,
            "max_delay": 0.0,
            "fraction_over_0_5s": 0.0,
            "fraction_over_1_0s": 0.0,
            "fraction_over_2_0s": 0.0,
            "fraction_over_slot": 0.0,
        }
    return {
        "count": int(len(delays)),
        "mean_delay": float(np.mean(delays)),
        "median_delay": float(np.median(delays)),
        "p90_delay": float(np.percentile(delays, 90)),
        "p95_delay": float(np.percentile(delays, 95)),
        "p99_delay": float(np.percentile(delays, 99)),
        "max_delay": float(np.max(delays)),
        "fraction_over_0_5s": float(np.mean(delays > 0.5)),
        "fraction_over_1_0s": float(np.mean(delays > 1.0)),
        "fraction_over_2_0s": float(np.mean(delays > 2.0)),
        "fraction_over_slot": float(np.mean(delays > float(slot_duration))),
    }


def summarize_manager_switching(
    manager_transitions,
    num_uavs,
    num_entities,
    uav_switches,
    movement_distances,
    dt_switches,
):
    manager_transitions = max(int(manager_transitions), 1)
    total_uav_switches = int(np.sum(np.asarray(uav_switches, dtype=np.int32)))
    total_dt_switches = int(np.sum(np.asarray(dt_switches, dtype=np.int32)))
    total_movement_distance = float(np.sum(np.asarray(movement_distances, dtype=np.float32)))

    return {
        "manager_transitions": int(manager_transitions),
        "uav_switch_fraction": float(total_uav_switches / max(manager_transitions * num_uavs, 1)),
        "dt_host_switch_fraction": float(total_dt_switches / max(manager_transitions * num_entities, 1)),
        "raw_uav_position_change_count": int(total_uav_switches),
        "raw_dt_host_change_count": int(total_dt_switches),
        "avg_changed_uavs_per_transition": float(total_uav_switches / manager_transitions),
        "avg_rehosted_entities_per_transition": float(total_dt_switches / manager_transitions),
        "total_grid_movement_distance": float(total_movement_distance),
        "avg_movement_distance_per_uav_transition": float(
            total_movement_distance / max(manager_transitions * num_uavs, 1)
        ),
    }
