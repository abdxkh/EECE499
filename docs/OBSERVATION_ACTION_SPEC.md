# Observation and Action Specification

## Worker Observation

- `Q`: shape `(I,)`, range `{0,1}`
- `U / aoi_obs_norm`: shape `(I,)`, nonnegative
- `W / packet_size_max`: shape `(I,)`, `[0,1]`
- `sensor_aoi / aoi_obs_norm`: shape `(I,)`, nonnegative
- `entity_aodt / aoi_obs_norm`: shape `(E,)`, nonnegative
- `sensor_uav_distances / area_diagonal`: shape `(I*M,)`, nonnegative
- `dt_host_one_hot`: shape `(E*M,)`, binary
- `sensor_entity_one_hot`: shape `(I*E,)`, binary
- `sensor_dt_host_one_hot`: shape `(I*M,)`, binary
- `backhaul_powers / backhaul_power_max`: shape `(M,)`, `[0,1]`

Default corrected dimension: `248`

## Worker Action

- `sensor_action[m]`:
  categorical over `I + 1` values
- real sensors `0..I-1`
- idle action `I`
- `power_action[m]`:
  - `learned_beta`: normalized Beta sample in `[0,1]`, mapped to `[p_min, p_max]`
  - `fixed_max`: implicit constant `p_max`
  - `fixed_mid`: implicit constant `(p_min + p_max)/2`

Executed environment action:

```text
[(sensor_id_or_minus_one, physical_power), ...]
```

## Manager Observation

- `time / episode_slots`: shape `(1,)`
- `uav_positions / area_size`: shape `(2M,)`
- `dt_host_one_hot`: shape `(E*M,)`
- `storage_used / uav_storage_capacity`: shape `(M,)`
- `dt_storage / max_uav_capacity`: shape `(E,)`
- `uav_storage_capacity / max_uav_capacity`: shape `(M,)`
- `entity_aodt / aoi_obs_norm`: shape `(E,)`
- `virtual_queues / energy_budget`: shape `(M,)`
- `last_window_energy / energy_budget`: shape `(M,)`
- `backhaul_powers / backhaul_power_max`: shape `(M,)`

Default corrected dimension: `47`

## Manager Action

- `uav_grid_indices`: shape `(M,)`, categorical per UAV
- `dt_assignment_index`: shape `(1,)`, categorical over all complete DT-host assignments
- `backhaul_powers`: shape `(M,)`, continuous bounded powers in `[backhaul_power_min, backhaul_power_max]`

Decoded DT assignment:

- `dt_hosts`: shape `(E,)`
- decoded exactly from `dt_assignment_index`
- must belong to the per-scenario feasible assignment subset

Legacy host mode:

- `dt_hosts` sampled independently per entity, then repaired
- kept only for explicit backward comparison
