import itertools
import hashlib
import numpy as np

from dt_uav_v2.config import CONFIG


class BaseUAVAoDTEnv:
    """
    This is the base simulator for the UAV Digital Twin AoDT problem.

    Important:
    - This file does NOT contain PPO.
    - This file does NOT contain neural networks.
    - This file only simulates the system behavior:
        sensors,
        UAVs,
        DT placement,
        packet arrivals,
        buffers,
        uplink delay,
        backhaul delay,
        processing delay,
        AoI / AoDT update,
        backhaul energy.
    """

    def __init__(self, config=None):
        """
        Constructor.

        We load all parameters from config.py and store them inside the environment.
        Nothing dynamic is initialized here yet.
        The actual episode state is initialized inside reset().
        """

        # If no custom config is given, use the default CONFIG dictionary.
        self.config = config or CONFIG

        # Random generator.
        # Using numpy default_rng makes results repeatable when we use the same seed.
        self.rng = np.random.default_rng(self.config["seed"])

        # -------------------------
        # Main system dimensions
        # -------------------------

        self.M = self.config["num_uavs"]          # number of UAVs
        self.E = self.config["num_entities"]     # number of physical entities / DTs
        self.I = self.config["num_sensors"]      # number of sensors

        # -------------------------
        # Time settings
        # -------------------------

        self.t = 0  # current time slot, initialized properly in reset()

        self.episode_slots = self.config["episode_slots"]  # total slots in one episode

        self.slot_duration = self.config["slot_duration"]  # slot duration in seconds
        self.service_model = self.config.get("service_model", "abstract_same_step")

        # -------------------------
        # Area settings
        # -------------------------

        self.area_size = self.config["area_size"]  # square area: [0, area_size] x [0, area_size]

        # -------------------------
        # Arrival / packet settings
        # -------------------------

        self.arrival_prob = self.config["arrival_prob"]  # probability of update arrival per sensor per slot

        self.packet_size_min = self.config["packet_size_min"]  # minimum packet size in bits

        self.packet_size_max = self.config["packet_size_max"]  # maximum packet size in bits

        # -------------------------
        # Communication settings
        # -------------------------

        self.B_access = self.config["bandwidth_access"]  # sensor-to-UAV bandwidth in Hz

        self.B_backhaul = self.config["bandwidth_backhaul"]  # UAV-to-UAV backhaul bandwidth in Hz

        self.noise_power = self.config["noise_power"]  # noise power for rate calculation

        self.pathloss_ref = self.config["pathloss_ref"]  # reference channel gain

        self.pathloss_exp = self.config["pathloss_exp"]  # pathloss exponent

        self.sensor_power_levels = self.config["sensor_power_levels"]  # discrete sensor power levels

        self.backhaul_power = self.config["backhaul_power"]  # default UAV backhaul transmit power

        self.backhaul_power_min = self.config.get("backhaul_power_min", self.backhaul_power)

        self.backhaul_power_max = self.config.get("backhaul_power_max", self.backhaul_power)

        # -------------------------
        # Processing settings
        # -------------------------

        self.cpu_cycles_per_bit = self.config["cpu_cycles_per_bit"]  # cycles needed per bit

        self.cpu_rate = self.config["cpu_rate"]  # CPU cycles per second

        # -------------------------
        # Storage settings
        # -------------------------

        self.dt_storage_min = self.config["dt_storage_min"]  # min DT storage size

        self.dt_storage_max = self.config["dt_storage_max"]  # max DT storage size

        self.uav_storage_capacity_value = self.config["uav_storage_capacity"]  # capacity per UAV

        # -------------------------
        # Variables initialized in reset()
        # -------------------------

        self.sensor_positions = None       # shape: (I, 2)
        self.uav_positions = None          # shape: (M, 2)

        self.sensor_entity = None          # shape: (I,), sensor_entity[i] = entity of sensor i
        self.dt_hosts = None               # shape: (E,), dt_hosts[e] = UAV hosting entity e's DT

        self.dt_storage = None             # shape: (E,)
        self.uav_storage_capacity = None   # shape: (M,)

        self.Q = None                      # shape: (I,), pending packet flag
        self.W = None                      # shape: (I,), packet size in bits
        self.U = None                      # shape: (I,), waiting time in slots

        self.sensor_aoi = None             # shape: (I,), AoI for each sensor at its DT
        self.entity_aodt = None            # shape: (E,), AoDT for each entity

        self.last_backhaul_energy = None   # shape: (M,), backhaul energy consumed by each UAV in last slot

        self.backhaul_powers = None        # shape: (M,), current backhaul transmit power per source UAV

        self.arrival_schedule = None       # shape: (episode_slots, I), exogenous packet arrivals
        self.packet_size_schedule = None   # shape: (episode_slots, I), exogenous packet sizes
        self.current_scenario = None       # replayable scenario snapshot
        self._all_dt_assignments = None    # cached complete DT-host assignments
        self._all_dt_assignment_indices = None

    # ============================================================
    # Reset and initialization helpers
    # ============================================================

    def reset(self, seed=None, scenario=None):
        """
        Start a new episode.

        This creates:
        - random sensor positions,
        - random UAV positions,
        - sensor-to-entity mapping,
        - DT placement,
        - packet buffers,
        - AoI / AoDT values.

        Returns:
            dictionary state for debugging.
        """

        if scenario is None:
            scenario = self.make_scenario_snapshot(seed=seed)

        self.load_scenario_snapshot(scenario, seed=seed)

        return self.get_basic_state()

    def make_scenario_snapshot(self, seed=None):
        """
        Create a replayable scenario snapshot containing all exogenous randomness.
        """

        scenario_seed = self.config["seed"] if seed is None else seed
        rng = np.random.default_rng(scenario_seed)

        sensor_positions = rng.uniform(
            low=0.0,
            high=self.area_size,
            size=(self.I, 2),
        ).astype(np.float32)
        uav_positions = rng.uniform(
            low=0.0,
            high=self.area_size,
            size=(self.M, 2),
        ).astype(np.float32)
        sensor_entity = self._create_sensor_entity_mapping()
        dt_storage = rng.uniform(
            low=self.dt_storage_min,
            high=self.dt_storage_max,
            size=self.E,
        ).astype(np.float32)
        uav_storage_capacity = (
            np.ones(self.M, dtype=np.float32) * self.uav_storage_capacity_value
        )

        if self.config.get("manager_host_action_mode", "feasible_enum") == "legacy_repair":
            initial_dt_hosts = self._repair_dt_storage(
                rng.integers(low=0, high=self.M, size=self.E),
                dt_storage=dt_storage,
                uav_storage_capacity=uav_storage_capacity,
            ).astype(int)
        else:
            initial_dt_hosts = self.sample_random_feasible_dt_assignment(
                rng=rng,
                dt_storage=dt_storage,
                uav_storage_capacity=uav_storage_capacity,
            ).astype(int)

        arrival_schedule = (rng.random((self.episode_slots, self.I)) < self.arrival_prob).astype(
            np.float32
        )
        packet_size_schedule = rng.uniform(
            low=self.packet_size_min,
            high=self.packet_size_max,
            size=(self.episode_slots, self.I),
        ).astype(np.float32)
        packet_size_schedule = packet_size_schedule * arrival_schedule

        feasible_set_hash = self.feasible_assignment_hash(
            dt_storage=dt_storage,
            uav_storage_capacity=uav_storage_capacity,
        )

        return {
            "scenario_seed": int(scenario_seed),
            "sensor_positions": sensor_positions.copy(),
            "initial_uav_positions": uav_positions.copy(),
            "sensor_entity": sensor_entity.copy(),
            "dt_storage": dt_storage.copy(),
            "uav_storage_capacity": uav_storage_capacity.copy(),
            "initial_dt_hosts": initial_dt_hosts.copy(),
            "initial_backhaul_powers": (
                np.ones(self.M, dtype=np.float32) * self.backhaul_power
            ),
            "arrival_schedule": arrival_schedule.copy(),
            "packet_size_schedule": packet_size_schedule.copy(),
            "feasible_assignment_hash": feasible_set_hash,
        }

    def load_scenario_snapshot(self, scenario, seed=None):
        """
        Load a replayable scenario snapshot into the environment state.
        """

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(int(scenario.get("scenario_seed", self.config["seed"])))

        self.current_scenario = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in scenario.items()
        }

        self.t = 0
        self.sensor_positions = np.asarray(scenario["sensor_positions"], dtype=np.float32).copy()
        self.uav_positions = np.asarray(scenario["initial_uav_positions"], dtype=np.float32).copy()
        self.sensor_entity = np.asarray(scenario["sensor_entity"], dtype=int).copy()
        self.dt_storage = np.asarray(scenario["dt_storage"], dtype=np.float32).copy()
        self.uav_storage_capacity = np.asarray(
            scenario["uav_storage_capacity"],
            dtype=np.float32,
        ).copy()
        self.dt_hosts = np.asarray(scenario["initial_dt_hosts"], dtype=int).copy()
        self.backhaul_powers = np.asarray(
            scenario.get(
                "initial_backhaul_powers",
                np.ones(self.M, dtype=np.float32) * self.backhaul_power,
            ),
            dtype=np.float32,
        ).copy()
        self.arrival_schedule = np.asarray(scenario["arrival_schedule"], dtype=np.float32).copy()
        self.packet_size_schedule = np.asarray(
            scenario["packet_size_schedule"],
            dtype=np.float32,
        ).copy()

        self.Q = np.zeros(self.I, dtype=np.float32)
        self.W = np.zeros(self.I, dtype=np.float32)
        self.U = np.zeros(self.I, dtype=np.float32)
        self.sensor_aoi = np.zeros(self.I, dtype=np.float32)
        self.entity_aodt = np.zeros(self.E, dtype=np.float32)
        self.last_backhaul_energy = np.zeros(self.M, dtype=np.float32)

        served_prev = np.zeros(self.I, dtype=np.float32)
        arrivals, packet_sizes = self.arrivals_for_slot(0)
        self.update_buffers(arrivals, packet_sizes, served_prev)
        self.update_entity_aodt()

    def _create_sensor_entity_mapping(self):
        """
        Create sensor-to-entity mapping.

        We want:
            sensor_entity[i] = e

        Simple version:
        - distribute sensors almost evenly across entities.
        - Example with I=15, E=5:
            sensors 0,1,2 -> entity 0
            sensors 3,4,5 -> entity 1
            etc.

        Returns:
            numpy array of shape (I,)
        """

        sensor_entity = np.zeros(self.I, dtype=int)

        for i in range(self.I):
            # This distributes sensors across entities in order.
            sensor_entity[i] = i % self.E

        return sensor_entity

    def _create_initial_dt_placement(self):
        """
        Create initial DT placement.

        dt_hosts[e] = m means:
            entity e's digital twin is hosted on UAV m.

        For now:
        - assign each entity randomly to a UAV,
        - then check storage,
        - if storage is violated, try a simple repair.

        Returns:
            numpy array of shape (E,)
        """

        if self.config.get("manager_host_action_mode", "feasible_enum") == "legacy_repair":
            dt_hosts = self.rng.integers(low=0, high=self.M, size=self.E)
            return self._repair_dt_storage(dt_hosts)

        return self.sample_random_feasible_dt_assignment(self.rng)

    def _repair_dt_storage(self, dt_hosts, dt_storage=None, uav_storage_capacity=None):
        """
        Repair DT placement if any UAV exceeds storage capacity.

        This is a simple repair method:
        - compute used storage per UAV,
        - if one UAV is overloaded,
        - move one DT from that UAV to another UAV that has enough space.

        This is not an optimization algorithm.
        It only makes sure the initial placement is valid enough for simulation.
        """

        dt_hosts = np.asarray(dt_hosts, dtype=int).copy()
        dt_storage = self.dt_storage if dt_storage is None else np.asarray(dt_storage, dtype=float)
        uav_storage_capacity = (
            self.uav_storage_capacity
            if uav_storage_capacity is None
            else np.asarray(uav_storage_capacity, dtype=float)
        )

        for _ in range(100):  # small safety loop to avoid infinite repair
            used = self.compute_storage_used(dt_hosts, dt_storage=dt_storage)

            overloaded_uavs = np.where(used > uav_storage_capacity)[0]

            # If no overloaded UAV, placement is valid.
            if len(overloaded_uavs) == 0:
                return dt_hosts

            overloaded = overloaded_uavs[0]

            # Find entities currently hosted on the overloaded UAV.
            entities_on_overloaded = np.where(dt_hosts == overloaded)[0]

            if len(entities_on_overloaded) == 0:
                return dt_hosts

            # Pick one entity to move.
            e = entities_on_overloaded[0]

            moved = False

            for target_uav in range(self.M):
                if target_uav == overloaded:
                    continue

                # Check if target has enough remaining capacity.
                if used[target_uav] + dt_storage[e] <= uav_storage_capacity[target_uav]:
                    dt_hosts[e] = target_uav
                    moved = True
                    break

            # If no move was possible, return as is.
            # In our config, this should not normally happen.
            if not moved:
                return dt_hosts

        return dt_hosts

    def compute_storage_used(self, dt_hosts=None, dt_storage=None):
        """
        Compute used storage per UAV.

        Args:
            dt_hosts:
                optional placement array.
                If None, use self.dt_hosts.

        Returns:
            used storage per UAV, shape (M,)
        """

        if dt_hosts is None:
            dt_hosts = self.dt_hosts

        dt_storage = self.dt_storage if dt_storage is None else np.asarray(dt_storage, dtype=float)
        used = np.zeros(self.M, dtype=np.float32)

        for e in range(self.E):
            host = dt_hosts[e]
            used[host] += dt_storage[e]

        return used

    def enumerate_all_dt_assignments(self):
        """
        Enumerate every complete DT-host assignment in a deterministic order.
        """

        if self._all_dt_assignments is None:
            assignments = list(itertools.product(range(self.M), repeat=self.E))
            self._all_dt_assignments = np.asarray(assignments, dtype=int)
            self._all_dt_assignment_indices = np.arange(len(assignments), dtype=int)

        return self._all_dt_assignments.copy()

    def feasible_dt_assignment_mask(self, dt_storage=None, uav_storage_capacity=None):
        """
        Return a boolean feasibility mask over all complete DT-host assignments.
        """

        all_assignments = self.enumerate_all_dt_assignments()
        dt_storage = self.dt_storage if dt_storage is None else np.asarray(dt_storage, dtype=float)
        uav_storage_capacity = (
            self.uav_storage_capacity
            if uav_storage_capacity is None
            else np.asarray(uav_storage_capacity, dtype=float)
        )

        used = np.zeros((len(all_assignments), self.M), dtype=np.float32)
        for e in range(self.E):
            used[np.arange(len(all_assignments)), all_assignments[:, e]] += dt_storage[e]

        return np.all(used <= uav_storage_capacity[None, :] + 1e-9, axis=1)

    def get_feasible_dt_assignments(self, dt_storage=None, uav_storage_capacity=None):
        mask = self.feasible_dt_assignment_mask(
            dt_storage=dt_storage,
            uav_storage_capacity=uav_storage_capacity,
        )
        return self.enumerate_all_dt_assignments()[mask]

    def get_feasible_dt_assignment_indices(self, dt_storage=None, uav_storage_capacity=None):
        if self._all_dt_assignment_indices is None:
            self.enumerate_all_dt_assignments()
        mask = self.feasible_dt_assignment_mask(
            dt_storage=dt_storage,
            uav_storage_capacity=uav_storage_capacity,
        )
        return self._all_dt_assignment_indices[mask].copy()

    def feasible_assignment_hash(self, dt_storage=None, uav_storage_capacity=None):
        feasible_indices = self.get_feasible_dt_assignment_indices(
            dt_storage=dt_storage,
            uav_storage_capacity=uav_storage_capacity,
        )
        digest = hashlib.sha256(feasible_indices.astype(np.int32).tobytes()).hexdigest()
        return digest

    def sample_random_feasible_dt_assignment(
        self,
        rng=None,
        dt_storage=None,
        uav_storage_capacity=None,
    ):
        """
        Sample one exactly feasible complete DT-host assignment.
        """

        rng = self.rng if rng is None else rng
        feasible = self.get_feasible_dt_assignments(
            dt_storage=dt_storage,
            uav_storage_capacity=uav_storage_capacity,
        )
        if len(feasible) == 0:
            raise RuntimeError("No feasible DT-host assignment exists for the current storage setting.")

        idx = int(rng.integers(low=0, high=len(feasible)))
        return feasible[idx].copy()

    def apply_manager_context(self, uav_positions, dt_hosts, backhaul_powers):
        """
        Apply a valid manager context exactly, with no hidden repair or clipping.
        """

        uav_positions = np.asarray(uav_positions, dtype=np.float32)
        dt_hosts = np.asarray(dt_hosts, dtype=int)
        backhaul_powers = np.asarray(backhaul_powers, dtype=np.float32)

        if uav_positions.shape != (self.M, 2):
            raise ValueError("uav_positions must have shape (num_uavs, 2).")
        if dt_hosts.shape != (self.E,):
            raise ValueError("dt_hosts must provide one host per entity.")
        if backhaul_powers.shape != (self.M,):
            raise ValueError("backhaul_powers must provide one value per UAV.")
        if np.any(dt_hosts < 0) or np.any(dt_hosts >= self.M):
            raise ValueError("dt_hosts contains an invalid UAV index.")
        if np.any(backhaul_powers < self.backhaul_power_min - 1e-9) or np.any(
            backhaul_powers > self.backhaul_power_max + 1e-9
        ):
            raise ValueError("backhaul_powers must lie within configured physical bounds.")
        if not np.all(self.compute_storage_used(dt_hosts) <= self.uav_storage_capacity + 1e-9):
            raise ValueError("dt_hosts must satisfy UAV storage capacities exactly.")

        self.uav_positions = uav_positions.copy()
        self.dt_hosts = dt_hosts.copy()
        self.backhaul_powers = backhaul_powers.copy()

    # ============================================================
    # Arrival and buffer logic
    # ============================================================

    def generate_arrivals(self):
        """
        Generate packet arrivals for the current slot.

        arrivals[i] = 1 means sensor i generated a fresh update.
        packet_sizes[i] = size of the generated packet in bits.

        Since arrival_prob can be 1.0, this supports both:
        - continuous monitoring: every sensor updates every slot,
        - random monitoring: each sensor updates with probability arrival_prob.

        Returns:
            arrivals, packet_sizes
        """

        arrivals = self.rng.random(self.I) < self.arrival_prob
        arrivals = arrivals.astype(float)

        packet_sizes = self.rng.uniform(
            low=self.packet_size_min,
            high=self.packet_size_max,
            size=self.I
        )

        # If no arrival, packet size should be 0.
        packet_sizes = packet_sizes * arrivals

        return arrivals, packet_sizes

    def arrivals_for_slot(self, slot_index):
        """
        Return the exogenous arrivals for a specific slot in the loaded scenario.
        """

        if self.arrival_schedule is None or self.packet_size_schedule is None:
            raise RuntimeError("No scenario arrival schedule is loaded.")
        return (
            self.arrival_schedule[slot_index].copy(),
            self.packet_size_schedule[slot_index].copy(),
        )

    def update_buffers(self, arrivals, packet_sizes, served_prev):
        """
        Update keep-the-latest buffers.

        For each sensor:
        - if a new packet arrives, it overwrites the old packet,
        - if no new packet arrives and previous packet was served, buffer becomes empty,
        - if no new packet arrives and previous packet was not served, old packet remains and waiting time increases.

        Args:
            arrivals:
                A_i(t), shape (I,)
            packet_sizes:
                D_i(t), shape (I,)
            served_prev:
                served_prev[i] = 1 if sensor i was served in previous slot.

        Updates:
            self.Q
            self.W
            self.U
        """

        new_Q = np.zeros(self.I)
        new_W = np.zeros(self.I)
        new_U = np.zeros(self.I)

        for i in range(self.I):
            if arrivals[i] == 1:
                # New fresh update arrives.
                # Keep-the-latest rule: overwrite old packet.
                new_Q[i] = 1
                new_W[i] = packet_sizes[i]
                new_U[i] = 0

            else:
                # No new update arrived.
                if self.Q[i] == 1 and served_prev[i] == 0:
                    # Old packet still pending.
                    new_Q[i] = 1
                    new_W[i] = self.W[i]
                    new_U[i] = self.U[i] + 1

                else:
                    # Either buffer was empty, or old packet was served.
                    new_Q[i] = 0
                    new_W[i] = 0
                    new_U[i] = 0

        self.Q = new_Q
        self.W = new_W
        self.U = new_U

    # ============================================================
    # Distance, channel, rate, and delay helpers
    # ============================================================

    def compute_sensor_uav_distances(self):
        """
        Compute distance from every sensor to every UAV.

        Returns:
            distances[i, m] = distance between sensor i and UAV m.
            Shape: (I, M)
        """

        distances = np.zeros((self.I, self.M))

        for i in range(self.I):
            for m in range(self.M):
                distances[i, m] = np.linalg.norm(
                    self.sensor_positions[i] - self.uav_positions[m]
                )

        return distances

    def compute_uav_uav_distances(self):
        """
        Compute distance from every UAV to every other UAV.

        Returns:
            distances[m, k] = distance between UAV m and UAV k.
            Shape: (M, M)
        """

        distances = np.zeros((self.M, self.M))

        for m in range(self.M):
            for k in range(self.M):
                distances[m, k] = np.linalg.norm(
                    self.uav_positions[m] - self.uav_positions[k]
                )

        return distances

    def channel_gain(self, distance):
        """
        Compute channel gain using a simple pathloss model.

        Formula:
            h = pathloss_ref / (distance + 1)^pathloss_exp

        We add +1 to avoid division by zero when distance is very small.
        """

        return self.pathloss_ref / ((distance + 1.0) ** self.pathloss_exp)

    def uplink_rate(self, sensor_id, uav_id, power):
        """
        Compute uplink rate from sensor i to UAV m.

        Formula:
            R = B_access * log2(1 + p*h/noise)

        Args:
            sensor_id: sensor index i
            uav_id: UAV index m
            power: sensor transmit power in watts

        Returns:
            uplink rate in bits per second
        """

        distance = np.linalg.norm(
            self.sensor_positions[sensor_id] - self.uav_positions[uav_id]
        )

        h = self.channel_gain(distance)

        snr = (power * h) / self.noise_power

        rate = self.B_access * np.log2(1.0 + snr)

        # Avoid impossible zero rate.
        return max(rate, 1e-9)

    def backhaul_rate(self, from_uav, to_uav, power=None):
        """
        Compute backhaul rate from UAV m to UAV k.

        For now, we split backhaul bandwidth equally among all directed UAV-to-UAV links.

        Number of directed links:
            M * (M - 1)

        Formula:
            R = B_link * log2(1 + p_bh*h/noise)

        Args:
            from_uav: transmitting UAV m
            to_uav: receiving UAV k

        Returns:
            backhaul rate in bits per second
        """

        if from_uav == to_uav:
            # No backhaul needed if source and destination are the same UAV.
            return 1e18

        num_links = self.M * (self.M - 1)

        B_link = self.B_backhaul / num_links

        distance = np.linalg.norm(
            self.uav_positions[from_uav] - self.uav_positions[to_uav]
        )

        h = self.channel_gain(distance)

        if power is None:
            power = self.backhaul_powers[from_uav]

        snr = (power * h) / self.noise_power

        rate = B_link * np.log2(1.0 + snr)

        return max(rate, 1e-9)

    def processing_delay(self, sensor_id):
        """
        Compute processing delay for a sensor update at the DT host.

        Formula:
            tau_proc = packet_size * cpu_cycles_per_bit / cpu_rate

        Args:
            sensor_id: sensor index i

        Returns:
            processing delay in seconds
        """

        packet_size = self.W[sensor_id]

        delay = (packet_size * self.cpu_cycles_per_bit) / self.cpu_rate

        return delay

    # ============================================================
    # One worker slot step
    # ============================================================

    def step_worker(self, action):
        """
        Apply one slot-level worker action.

        Action format:
            action = [
                (sensor_for_uav_0, power_index_for_uav_0),
                (sensor_for_uav_1, power_index_for_uav_1),
                ...
            ]

        Example with 3 UAVs:
            [
                (2, 1),   # UAV 0 serves sensor 2 using power_levels[1]
                (5, 3),   # UAV 1 serves sensor 5 using power_levels[3]
                (-1, 0),  # UAV 2 stays idle
            ]

        sensor_id = -1 means:
            UAV stays idle.

        What this function does:
        - checks invalid actions,
        - computes delays,
        - computes backhaul energy,
        - updates AoI / AoDT,
        - generates arrivals for next slot,
        - updates buffers for next slot,
        - returns debug info.
        """

        # Track which sensors were completed successfully in this slot.
        served = np.zeros(self.I, dtype=np.float32)

        # Track all valid transmission attempts separately from successful completions.
        attempted = np.zeros(self.I, dtype=np.float32)
        completed = np.zeros(self.I, dtype=np.float32)

        # Delay components for each sensor attempt.
        uplink_delay = np.zeros(self.I, dtype=np.float32)
        backhaul_delay = np.zeros(self.I, dtype=np.float32)
        processing_delay = np.zeros(self.I, dtype=np.float32)
        total_delay = np.zeros(self.I, dtype=np.float32)
        uplink_rate = np.zeros(self.I, dtype=np.float32)
        backhaul_rate = np.zeros(self.I, dtype=np.float32)
        selected_uplink_power = np.zeros(self.I, dtype=np.float32)
        selected_backhaul_power = np.zeros(self.I, dtype=np.float32)
        sensor_uav_distance = np.zeros(self.I, dtype=np.float32)
        uav_host_distance = np.zeros(self.I, dtype=np.float32)
        packet_sizes = np.zeros(self.I, dtype=np.float32)
        direct_upload = np.zeros(self.I, dtype=np.float32)
        transmission_records = []

        # Backhaul energy consumed by each UAV in this slot.
        backhaul_energy = np.zeros(self.M)

        # Debug counters.
        invalid_count = 0
        wasted_count = 0

        # Used to prevent two UAVs from serving the same sensor in the same slot.
        selected_sensors = set()

        # Make sure the action length matches number of UAVs.
        if len(action) != self.M:
            raise ValueError("Action length must equal number of UAVs.")

        # --------------------------------------------------------
        # Apply each UAV action
        # --------------------------------------------------------

        for m in range(self.M):
            sensor_id, power_index = action[m]

            # -------------------------
            # Case 1: UAV stays idle
            # -------------------------
            if sensor_id == -1:
                wasted_count += 1
                continue

            # -------------------------
            # Case 2: invalid sensor index
            # -------------------------
            if sensor_id < 0 or sensor_id >= self.I:
                invalid_count += 1
                continue

            # -------------------------
            # Case 3: invalid power choice
            # -------------------------
            if self.config.get("worker_continuous_power", False):
                if not np.isfinite(power_index):
                    invalid_count += 1
                    continue
            else:
                if power_index < 0 or power_index >= len(self.sensor_power_levels):
                    invalid_count += 1
                    continue

            # -------------------------
            # Case 4: sensor has no pending packet
            # -------------------------
            if self.Q[sensor_id] == 0:
                invalid_count += 1
                continue

            # -------------------------
            # Case 5: same sensor selected by another UAV
            # -------------------------
            if sensor_id in selected_sensors:
                invalid_count += 1
                continue

            selected_sensors.add(sensor_id)
            attempted[sensor_id] = 1.0

            # Power chosen by the worker. In discrete mode, power_index is an
            # index. In continuous mode, it is the actual transmit power.
            if self.config.get("worker_continuous_power", False):
                power = float(power_index)
                if power < min(self.sensor_power_levels) - 1e-9 or power > max(self.sensor_power_levels) + 1e-9:
                    raise ValueError("Continuous worker power lies outside configured physical bounds.")
            else:
                power = self.sensor_power_levels[power_index]

            # Packet size to transmit.
            packet_size = self.W[sensor_id]
            packet_sizes[sensor_id] = packet_size
            selected_uplink_power[sensor_id] = power

            # -------------------------
            # Uplink delay
            # -------------------------

            sensor_distance = float(
                np.linalg.norm(self.sensor_positions[sensor_id] - self.uav_positions[m])
            )
            R_ul = self.uplink_rate(sensor_id, m, power)

            tau_ul = packet_size / R_ul
            sensor_uav_distance[sensor_id] = sensor_distance
            uplink_rate[sensor_id] = R_ul
            uplink_delay[sensor_id] = tau_ul

            # -------------------------
            # Backhaul delay and energy
            # -------------------------

            entity_id = self.sensor_entity[sensor_id]

            dt_host = self.dt_hosts[entity_id]

            tau_bh = 0.0
            e_bh = 0.0
            bh_power = 0.0
            R_bh = 0.0
            host_distance = 0.0

            # If the serving UAV does not host the sensor's entity DT,
            # then forwarding over backhaul is required.
            if m != dt_host:
                bh_power = self.backhaul_powers[m]

                R_bh = self.backhaul_rate(m, dt_host, power=bh_power)
                host_distance = float(
                    np.linalg.norm(self.uav_positions[m] - self.uav_positions[dt_host])
                )

                tau_bh = packet_size / R_bh

                e_bh = bh_power * tau_bh

                # Energy is charged to the forwarding/source UAV.
                backhaul_energy[m] += e_bh
            else:
                direct_upload[sensor_id] = 1.0

            selected_backhaul_power[sensor_id] = bh_power
            backhaul_rate[sensor_id] = R_bh
            backhaul_delay[sensor_id] = tau_bh
            uav_host_distance[sensor_id] = host_distance

            # -------------------------
            # Processing delay
            # -------------------------

            tau_proc = self.processing_delay(sensor_id)
            processing_delay[sensor_id] = tau_proc

            # -------------------------
            # Total end-to-end delay
            # -------------------------

            delay = tau_ul + tau_bh + tau_proc
            total_delay[sensor_id] = delay
            completion_success = (
                True
                if self.service_model == "abstract_same_step"
                else bool(delay <= self.slot_duration + 1e-9)
            )
            if self.service_model not in {"abstract_same_step", "require_within_slot"}:
                raise ValueError(f"Unknown service model: {self.service_model}")

            if completion_success:
                served[sensor_id] = 1.0
                completed[sensor_id] = 1.0

            transmission_records.append(
                {
                    "time_slot": int(self.t),
                    "sensor_id": int(sensor_id),
                    "uav_id": int(m),
                    "entity_id": int(entity_id),
                    "dt_host_uav": int(dt_host),
                    "direct_upload": bool(m == dt_host),
                    "cross_upload": bool(m != dt_host),
                    "attempted": True,
                    "completed": bool(completion_success),
                    "selected_uplink_power": float(power),
                    "selected_backhaul_power": float(bh_power),
                    "sensor_uav_distance": float(sensor_distance),
                    "uav_host_distance": float(host_distance),
                    "packet_size_bits": float(packet_size),
                    "uplink_rate_bps": float(R_ul),
                    "backhaul_rate_bps": float(R_bh),
                    "uplink_delay_s": float(tau_ul),
                    "backhaul_delay_s": float(tau_bh),
                    "processing_delay_s": float(tau_proc),
                    "total_delay_s": float(delay),
                }
            )

        # --------------------------------------------------------
        # Update AoI after serving decisions
        # --------------------------------------------------------

        self.update_sensor_aoi(served, total_delay)

        self.update_entity_aodt()

        # Save last backhaul energy for logs / manager.
        self.last_backhaul_energy = backhaul_energy

        # Average AoDT after this slot.
        avg_aodt = self.average_aodt()

        # --------------------------------------------------------
        # Advance time
        # --------------------------------------------------------

        self.t += 1

        done = self.t >= self.episode_slots

        # --------------------------------------------------------
        # Generate next slot arrivals and update buffers
        # --------------------------------------------------------
        # Important:
        # We update buffers after AoI update.
        # This means current slot used the packets that existed at beginning of slot.
        # Then new arrivals prepare the next slot.
        # --------------------------------------------------------

        if not done:
            arrivals, packet_sizes = self.arrivals_for_slot(self.t)
            self.update_buffers(arrivals, packet_sizes, served)

        # --------------------------------------------------------
        # Return debug information
        # --------------------------------------------------------

        info = {
            "time": self.t,
            "served": served,
            "attempted": attempted,
            "completed": completed,
            "total_delay": total_delay,
            "uplink_delay": uplink_delay,
            "backhaul_delay": backhaul_delay,
            "processing_delay": processing_delay,
            "uplink_rate": uplink_rate,
            "backhaul_rate": backhaul_rate,
            "selected_uplink_power": selected_uplink_power,
            "selected_backhaul_power": selected_backhaul_power,
            "sensor_uav_distance": sensor_uav_distance,
            "uav_host_distance": uav_host_distance,
            "packet_size_bits": packet_sizes,
            "direct_upload": direct_upload,
            "transmission_records": transmission_records,
            "completion_rate": float(np.mean(completed[attempted > 0.5])) if np.any(attempted > 0.5) else 0.0,
            "service_model": self.service_model,
            "backhaul_energy": backhaul_energy,
            "invalid_count": invalid_count,
            "wasted_count": wasted_count,
            "avg_aodt": avg_aodt,
            "entity_aodt": self.entity_aodt.copy(),
            "sensor_aoi": self.sensor_aoi.copy(),
            "done": done,
        }

        return self.get_basic_state(), info

    # ============================================================
    # AoI / AoDT helpers
    # ============================================================

    def update_sensor_aoi(self, served, total_delay):
        """
        Update sensor-level AoI.

        For each sensor:
        - if served:
            AoI becomes waiting_time + normalized_delay
        - if not served:
            AoI increases by 1

        Formula:
            Delta_i(t+1) = U_i(t) + d_i(t)/slot_duration      if served
            Delta_i(t+1) = Delta_i(t) + 1                     if not served
        """

        for i in range(self.I):
            if served[i] == 1:
                normalized_delay = total_delay[i] / self.slot_duration
                self.sensor_aoi[i] = self.U[i] + normalized_delay
            else:
                self.sensor_aoi[i] += 1

    def update_entity_aodt(self):
        """
        Update entity-level AoDT.

        Entity AoDT is the maximum AoI among sensors belonging to that entity.

        Formula:
            Delta_e(t) = max Delta_i(t) for all i in N_e
        """

        for e in range(self.E):
            sensors_of_e = np.where(self.sensor_entity == e)[0]

            if len(sensors_of_e) == 0:
                self.entity_aodt[e] = 0
            else:
                self.entity_aodt[e] = np.max(self.sensor_aoi[sensors_of_e])

    def average_aodt(self):
        """
        Return average entity AoDT.
        """

        return float(np.mean(self.entity_aodt))

    def tail_aodt(self, percentile=95):
        """
        Return tail AoDT.

        This is useful later in evaluation.
        Example:
            95th percentile AoDT.
        """

        return float(np.percentile(self.entity_aodt, percentile))

    # ============================================================
    # State helper
    # ============================================================

    def get_basic_state(self):
        """
        Return a dictionary state.

        This is mainly for debugging.

        Later:
        - worker_env.py will convert this into worker neural network observation.
        - manager_env.py will convert this into manager neural network observation.
        """

        state = {
            "time": self.t,
            "sensor_positions": self.sensor_positions.copy(),
            "uav_positions": self.uav_positions.copy(),
            "sensor_entity": self.sensor_entity.copy(),
            "dt_hosts": self.dt_hosts.copy(),
            "dt_storage": self.dt_storage.copy(),
            "uav_storage_capacity": self.uav_storage_capacity.copy(),
            "Q": self.Q.copy(),
            "W": self.W.copy(),
            "U": self.U.copy(),
            "sensor_aoi": self.sensor_aoi.copy(),
            "entity_aodt": self.entity_aodt.copy(),
            "last_backhaul_energy": self.last_backhaul_energy.copy(),
            "backhaul_powers": self.backhaul_powers.copy(),
            "storage_used": self.compute_storage_used().copy(),
        }

        return state

    # ============================================================
    # Random action helper for debugging
    # ============================================================

    def sample_random_worker_action(self):
        """
        Create a random worker action.

        For each UAV:
        - with small probability, stay idle,
        - otherwise choose a random sensor,
        - choose a random power level.

        This is only for testing the simulator.
        PPO will later produce the action.
        """

        action = []

        for m in range(self.M):
            # 10% chance to stay idle.
            if self.rng.random() < 0.1:
                action.append((-1, 0))
                continue

            sensor_id = int(self.rng.integers(low=0, high=self.I))

            if self.config.get("worker_continuous_power", False):
                power_index = float(
                    self.rng.uniform(
                        low=min(self.sensor_power_levels),
                        high=max(self.sensor_power_levels),
                    )
                )
            else:
                power_index = int(self.rng.integers(low=0, high=len(self.sensor_power_levels)))

            action.append((sensor_id, power_index))

        return action


# ============================================================
# Debug test
# ============================================================

if __name__ == "__main__":
    """
    Run this file directly to test the environment.

    From project root, use:

        python -m src.dt_uav_v2.envs.base_env

    or if your PYTHONPATH is set to src:

        python -m dt_uav_v2.envs.base_env
    """

    env = BaseUAVAoDTEnv()

    state = env.reset()

    print("Environment reset successfully.")
    print("Number of UAVs:", env.M)
    print("Number of entities:", env.E)
    print("Number of sensors:", env.I)
    print("Initial average AoDT:", env.average_aodt())
    print("Initial DT hosts:", env.dt_hosts)
    print("Initial storage used:", env.compute_storage_used())
    print()

    for step in range(5):
        action = env.sample_random_worker_action()

        state, info = env.step_worker(action)

        print("Step:", step + 1)
        print("Action:", action)
        print("Served sensors:", np.where(info["served"] == 1)[0])
        print("Invalid actions:", info["invalid_count"])
        print("Wasted slots:", info["wasted_count"])
        print("Backhaul energy:", info["backhaul_energy"])
        print("Entity AoDT:", info["entity_aodt"])
        print("Average AoDT:", info["avg_aodt"])
        print("-" * 50)
