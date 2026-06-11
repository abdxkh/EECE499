import numpy as np

from dt_uav_v2.config import CONFIG
from dt_uav_v2.envs.worker_env import WorkerEnv


def make_debug_config():
    config = dict(CONFIG)
    config.update(
        {
            "seed": 7,
            "num_uavs": 2,
            "num_entities": 2,
            "num_sensors": 4,
            "area_size": 100.0,
            "episode_slots": 4,
            "manager_horizon": 2,
            "arrival_prob": 1.0,
            "packet_size_min": 8000.0,
            "packet_size_max": 8000.0,
            "dt_storage_min": 40.0,
            "dt_storage_max": 40.0,
            "uav_storage_capacity": 100.0,
            "aoi_obs_norm": 10.0,
        }
    )
    return config


def set_manual_debug_state(env):
    """
    Replace random reset values with a tiny deterministic topology.

    Layout:
    - 2 UAVs: UAV 0 at x=0, UAV 1 at x=100
    - 4 sensors: sensors 0 and 2 near UAV 0, sensors 1 and 3 near UAV 1
    - 2 entities: sensors 0 and 2 belong to entity 0, sensors 1 and 3 to entity 1
    - DT 0 hosted on UAV 0, DT 1 hosted on UAV 1
    """

    base = env.base_env

    base.sensor_positions = np.asarray(
        [
            [10.0, 10.0],
            [90.0, 10.0],
            [20.0, 20.0],
            [80.0, 20.0],
        ],
        dtype=float,
    )
    base.uav_positions = np.asarray(
        [
            [0.0, 0.0],
            [100.0, 0.0],
        ],
        dtype=float,
    )
    base.sensor_entity = np.asarray([0, 1, 0, 1], dtype=int)
    base.dt_hosts = np.asarray([0, 1], dtype=int)
    base.dt_storage = np.asarray([40.0, 40.0], dtype=float)
    base.uav_storage_capacity = np.asarray([100.0, 100.0], dtype=float)

    base.Q = np.ones(base.I, dtype=float)
    base.W = np.asarray([8000.0, 8000.0, 8000.0, 8000.0], dtype=float)
    base.U = np.zeros(base.I, dtype=float)
    base.sensor_aoi = np.zeros(base.I, dtype=float)
    base.entity_aodt = np.zeros(base.E, dtype=float)
    base.last_backhaul_energy = np.zeros(base.M, dtype=float)
    base.t = 0

    base.update_entity_aodt()
    env.obs_dim = len(env._state_to_obs(base.get_basic_state()))


def print_initial_state(env):
    base = env.base_env

    print("Small debug network")
    print("UAVs:", base.M)
    print("Entities:", base.E)
    print("Sensors:", base.I)
    print("Worker observation dim:", env.obs_dim)
    print()
    print("Sensor positions:")
    print(base.sensor_positions)
    print("UAV positions:")
    print(base.uav_positions)
    print("Sensor -> entity:", base.sensor_entity)
    print("Entity DT hosts:", base.dt_hosts)
    print("Packet flags Q:", base.Q)
    print("Packet sizes W:", base.W)
    print("Initial sensor AoI:", base.sensor_aoi)
    print("Initial entity AoDT:", base.entity_aodt)
    print("Sensor-UAV distances:")
    print(np.round(base.compute_sensor_uav_distances(), 3))
    print("-" * 70)


def describe_action_calculations(env, action):
    base = env.base_env

    print("Manual calculation before applying action:")
    for uav_id, (sensor_id, power_index) in enumerate(action):
        if sensor_id == -1:
            print(f"  UAV {uav_id}: idle")
            continue

        power = base.sensor_power_levels[power_index]
        entity_id = int(base.sensor_entity[sensor_id])
        dt_host = int(base.dt_hosts[entity_id])
        packet_size = float(base.W[sensor_id])

        uplink_rate = base.uplink_rate(sensor_id, uav_id, power)
        uplink_delay = packet_size / uplink_rate
        processing_delay = base.processing_delay(sensor_id)

        backhaul_delay = 0.0
        backhaul_energy = 0.0
        if uav_id != dt_host:
            backhaul_rate = base.backhaul_rate(uav_id, dt_host)
            backhaul_delay = packet_size / backhaul_rate
            backhaul_energy = base.backhaul_power * backhaul_delay

        total_delay = uplink_delay + backhaul_delay + processing_delay

        print(
            f"  UAV {uav_id} serves sensor {sensor_id}: "
            f"entity={entity_id}, dt_host=UAV {dt_host}, power={power:.2f} W"
        )
        print(
            f"    uplink_delay={uplink_delay:.6f}, "
            f"backhaul_delay={backhaul_delay:.6f}, "
            f"processing_delay={processing_delay:.6f}, "
            f"total_delay={total_delay:.6f}, "
            f"backhaul_energy={backhaul_energy:.6f}"
        )


def run_debug_episode():
    config = make_debug_config()
    env = WorkerEnv(config=config)
    env.reset(seed=config["seed"])
    set_manual_debug_state(env)
    print_initial_state(env)

    max_power_index = env.num_power_levels - 1
    actions = [
        [(0, max_power_index), (1, max_power_index)],
        [(2, max_power_index), (3, max_power_index)],
        [(1, max_power_index), (0, max_power_index)],
        [(3, max_power_index), (2, max_power_index)],
    ]

    for step_id, action in enumerate(actions, start=1):
        print(f"Step {step_id}")
        print("Action:", action)
        print("AoI before:", np.round(env.base_env.sensor_aoi, 6))
        print("AoDT before:", np.round(env.base_env.entity_aodt, 6))

        describe_action_calculations(env, action)
        obs, reward, done, info = env.step(action)

        print("Served sensors:", np.where(info["served"] == 1)[0])
        print("Total delay per sensor:", np.round(info["total_delay"], 6))
        print("Backhaul energy per UAV:", np.round(info["backhaul_energy"], 6))
        print("AoI after:", np.round(info["sensor_aoi"], 6))
        print("AoDT after:", np.round(info["entity_aodt"], 6))
        print("Average AoDT:", round(info["avg_aodt"], 6))
        print("Reward:", round(reward, 6))
        print("Next packet flags Q:", env.base_env.Q)
        print("Next waiting times U:", env.base_env.U)
        print("Done:", done)
        print("-" * 70)

        if done:
            break


if __name__ == "__main__":
    run_debug_episode()
