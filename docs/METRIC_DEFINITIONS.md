# Metric Definitions

## Delay Metrics

- uplink delay:
  `packet_size / uplink_rate`, seconds
- backhaul delay:
  `packet_size / backhaul_rate`, seconds
- processing delay:
  `packet_size * cpu_cycles_per_bit / cpu_rate`, seconds
- total end-to-end delay:
  uplink + backhaul + processing, seconds
- delay-over-slot fraction:
  fraction of served updates whose total end-to-end delay exceeds `slot_duration`

## Freshness Metrics

- sensor AoI:
  slot units
- served sensor AoI update:
  `waiting_time + total_delay / slot_duration`
- unserved sensor AoI update:
  previous AoI + 1
- entity AoDT:
  maximum AoI over sensors belonging to the entity
- average AoDT:
  arithmetic mean over entities
- tail AoDT:
  percentile over entity AoDT values

## Energy Metrics

- slot backhaul energy per UAV:
  joules, counted only when forwarding is needed
- window-average backhaul energy per UAV:
  arithmetic mean over worker slots in the manager window, joules per slot
- evaluation `mean_energy`:
  average across windows of the mean per-UAV window-average backhaul energy
- evaluation `max_energy`:
  average across windows of the maximum per-UAV window-average backhaul energy
- energy budget:
  same unit as window-average backhaul energy per UAV

## Constraint Metrics

- signed violation:
  `avg_energy_per_uav - energy_budget`
- positive violation:
  `max(signed_violation, 0)`
- violation rate:
  fraction of manager windows where any UAV exceeds the budget
- virtual queue:
  `Z_{m}(k+1) = max(0, Z_m(k) + E_bar_m(k) - E_max_m)`

## PPO/Internal Metrics

- reward:
  algorithmic objective value, not a physical metric
- worker invalid count:
  number of slot actions rejected by the simulator
- worker wasted count:
  number of idle UAV choices in a slot
- manager reward terms:
  normalized AoDT term, queue-weighted energy term, total reward

## Switching Diagnostics

- manager actions per episode:
  number of slow-timescale decisions
- UAV switch rate:
  average number of UAV grid changes per manager action
- mean movement distance:
  average total UAV displacement per manager action, meters
- DT switch rate:
  average number of entity DT-host changes per manager action
