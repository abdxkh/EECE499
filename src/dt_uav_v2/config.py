# src/dt_uav_v2/config.py

CONFIG = {
    # -------------------------
    # General setup
    # -------------------------

    "seed": 42,  # random seed so results can be repeated

    "num_uavs": 3,  # number of UAV edge servers

    "num_entities": 5,  # number of physical entities / digital twins

    "num_sensors": 15,  # total number of IoT sensors

    "area_size": 1000.0,  # square area size, example: 1000 x 1000 meters

    "slot_duration": 1.0,  # duration of one time slot in seconds

    "manager_horizon": 5,  # manager acts once every H worker slots in the final experiments

    "manager_grid_size": 4,  # manager chooses UAV hover points from a grid_size x grid_size grid

    "episode_slots": 300,  # total number of slots in one episode

    # -------------------------
    # Sensor update model
    # -------------------------

    "arrival_prob": 1.0,  # probability that each sensor generates an update each slot; 1.0 means always fresh update

    "packet_size_min": 5000.0,  # minimum generated packet size in bits

    "packet_size_max": 20000.0,  # maximum generated packet size in bits

    # -------------------------
    # Communication parameters
    # -------------------------

    "bandwidth_total": 1_000_000.0,  # total available bandwidth in Hz

    "bandwidth_access": 700_000.0,  # bandwidth used for sensor-to-UAV uplink in Hz

    "bandwidth_backhaul": 300_000.0,  # bandwidth used for UAV-to-UAV backhaul in Hz

    "noise_power": 1e-9,  # noise power used in rate calculation

    "pathloss_ref": 1e-3,  # reference channel gain at distance 1 meter

    "pathloss_exp": 2.2,  # pathloss exponent; larger value means signal weakens faster with distance

    "sensor_power_levels": [0.05, 0.1, 0.15, 0.2],  # available sensor transmit power levels in watts

    "worker_continuous_power": True,  # worker outputs continuous sensor power in [min, max]

    "worker_power_mode": "fixed_max",  # one of: learned_beta, fixed_max, fixed_mid

    "backhaul_power": 1.0,  # default UAV backhaul transmit power in watts

    "optimize_backhaul_power": True,  # manager outputs continuous backhaul power per transmitting UAV

    "backhaul_power_min": 0.1,  # minimum optimized UAV backhaul transmit power in watts

    "backhaul_power_max": 1.0,  # maximum optimized UAV backhaul transmit power in watts

    # -------------------------
    # Processing model
    # -------------------------

    "cpu_cycles_per_bit": 1000.0,  # CPU cycles needed to process one bit of update data

    "cpu_rate": 1e9,  # UAV CPU processing rate in cycles per second

    # -------------------------
    # DT storage model
    # -------------------------

    "dt_storage_min": 50.0,  # minimum storage required by one DT replica

    "dt_storage_max": 100.0,  # maximum storage required by one DT replica

    "uav_storage_capacity": 250.0,  # storage capacity of each UAV

    # -------------------------
    # Backhaul energy constraint
    # -------------------------

    "backhaul_energy_budget": 0.25,  # allowed average backhaul energy per UAV per manager window

    # -------------------------
    # Reward weights
    # -------------------------

    "aodt_reward_scale": 10.0,  # divide AoDT by this value in worker reward; smaller means stronger AoDT pressure

    "aodt_delta_weight": 2.0,  # reward improvements in AoDT and punish slot-to-slot AoDT increases

    "worker_reward_mode": "current",  # worker PPO reward used in the saved experiments

    "aoi_obs_norm": 20.0,  # normalize AoI/AoDT observations to a useful learning scale

    "worker_freshness_bias": 4.0,  # policy prior toward high-AoI sensors during worker sampling

    "worker_force_max_power": False,  # fixed_max mode handles the final fixed uplink power behavior

    "invalid_action_penalty": 0.05,  # tiny guardrail penalty; AoDT should dominate worker learning

    "wasted_slot_penalty": 0.01,  # tiny guardrail penalty; AoDT should dominate worker learning

    "lyapunov_beta": 5.0,  # weight of the Lyapunov virtual queue penalty in manager reward

    "manager_reward_mode": "queue_weighted_energy",  # one of: legacy_queue_penalty, queue_weighted_energy

    "manager_host_action_mode": "feasible_enum",  # one of: feasible_enum, legacy_repair

    "manager_aodt_weight": 20.0,  # V coefficient for normalized window AoDT in manager reward

    "manager_energy_weight": 1.0,  # lambda coefficient for queue-weighted normalized backhaul energy

    "worker_context_mode": "random_feasible_context",  # one of: fixed_context, random_feasible_context, heuristic_context

    "validation_worker_scenarios": 40,  # number of held-out validation scenarios for worker checkpointing

    "validation_manager_scenarios": 40,  # number of held-out validation scenarios for manager checkpointing

    "test_scenarios_default": 200,  # final shared test-scenario count

    "scenario_seed_offsets": {
        "train": 0,
        "validation": 10_000,
        "test": 20_000,
    },

    "obs_spec_version": "phase_b_v1",
    "env_version": "phase_b5_v1",
    "scenario_distribution_version": "phase_b5_v1",
    "service_model": "require_within_slot",  # one of: abstract_same_step, require_within_slot

    # -------------------------
    # PPO hyperparameters
    # We will tune these later
    # -------------------------

    "worker_lr": 3e-4,  # learning rate for worker PPO

    "manager_lr": 1e-4,  # learning rate for manager PPO in the selected final configuration

    "manager_entropy_coef": 0.01,  # entropy coefficient for the selected final manager PPO

    "gamma": 0.99,  # discount factor for future rewards

    "gae_lambda": 0.95,  # GAE parameter for advantage estimation

    "clip_eps": 0.2,  # PPO clipping range
}
