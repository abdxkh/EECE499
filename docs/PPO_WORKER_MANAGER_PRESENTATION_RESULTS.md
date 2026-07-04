# PPO Worker + PPO Manager Presentation Results

These plots are generated from saved Phase C2 final artifacts only. No training or evaluation was rerun.

## Systems Plotted

- `ppo_worker_candidate0`: PPO worker with PPO manager candidate 0, entropy coefficient 0.01.
- `ppo_worker_candidate1`: PPO worker with PPO manager candidate 1, entropy coefficient 0.005.

## Test Summary

| system | seeds | test_mean_aodt | test_std_aodt_across_seeds | test_p95_episode_aodt | test_mean_energy | test_violation_fraction | test_p95_delay |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO worker + PPO manager C0 | 5 | 5.5322 | 0.1121 | 5.7764 | 0.0474 | 0.0019 | 0.4423 |
| PPO worker + PPO manager C1 | 5 | 5.5185 | 0.1022 | 5.7873 | 0.0340 | 0.0000 | 0.3535 |

## Best PPO-Only Seed

- Best seed by test AoDT: `ppo_worker_candidate1`, seed `55` with mean AoDT `5.3520`, energy `0.0423`, and violation fraction `0.0000`.
- Candidate comparison: `ppo_worker_candidate0` minus `ppo_worker_candidate1` mean paired AoDT difference = `0.0137`, 95% CI `[0.0051, 0.0222]`, p-value `0.001945`.

## Generated Figures

- `ppo_manager_validation_aodt.png`: validation AoDT during PPO-manager training.
- `ppo_manager_validation_energy.png`: validation energy during PPO-manager training.
- `c1_aodt_improvement_over_time.png`: C1 validation AoDT and running-best AoDT over training.
- `c1_backhaul_energy_over_time.png`: C1 validation backhaul energy and energy budget over training.

## Presentation Note

This is the PPO-worker + PPO-manager result set. It is useful for explaining how PPO was implemented and evaluated, but it should be presented as the PPO-only hierarchical baseline rather than as the final selected controller, because the final selected controller uses the greedy worker with the PPO manager.
