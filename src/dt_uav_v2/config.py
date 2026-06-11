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

    "manager_horizon": 10,  # manager acts once every H slots

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

    "aoi_obs_norm": 20.0,  # normalize AoI/AoDT observations to a useful learning scale

    "worker_freshness_bias": 4.0,  # policy prior toward high-AoI sensors during worker sampling

    "worker_force_max_power": False,  # continuous worker power is learned by PPO

    "invalid_action_penalty": 0.05,  # tiny guardrail penalty; AoDT should dominate worker learning

    "wasted_slot_penalty": 0.01,  # tiny guardrail penalty; AoDT should dominate worker learning

    "lyapunov_beta": 5.0,  # weight of the Lyapunov virtual queue penalty in manager reward

    # -------------------------
    # PPO hyperparameters
    # We will tune these later
    # -------------------------

    "worker_lr": 3e-4,  # learning rate for worker PPO

    "manager_lr": 3e-4,  # learning rate for manager PPO

    "gamma": 0.99,  # discount factor for future rewards

    "gae_lambda": 0.95,  # GAE parameter for advantage estimation

    "clip_eps": 0.2,  # PPO clipping range
}
