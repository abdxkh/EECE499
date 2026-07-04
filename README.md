# DT-Assisted UAV Edge Network for AoDT Minimization

This repository contains my implementation of a DT-assisted UAV edge-network simulator and a hierarchical reinforcement-learning controller for minimizing Age of Digital Twin (AoDT). The project studies two timescales:

- a fast worker controller that acts every slot and decides which sensor each UAV serves;
- a slow manager controller that acts every manager window and decides UAV placement, DT hosting, and backhaul power.

The repository includes the full hierarchical PPO implementation and the later comparison runs used to understand the behavior of the controllers. The main evaluated setup is:

```text
fast controller: PPO worker baseline and greedy maximum-age-reduction worker comparison
slow controller: PPO manager
service model: within-slot service completion
uplink power: fixed maximum sensor power
slot duration: Ts = 1.0 s
manager horizon: H = 5 worker slots
```

The full PPO-worker + PPO-manager hierarchy is preserved and evaluated, and the greedy-worker + PPO-manager comparison is also included in the saved results.

## Repository Structure

```text
src/dt_uav_v2/
  config.py                    Main experiment parameters
  envs/base_env.py             Physical simulator, buffers, AoI/AoDT, delay, energy
  envs/worker_env.py           Worker MDP wrapper and greedy worker
  envs/manager_env.py          Manager MDP wrapper and virtual queues
  agents/ppo.py                PPO actor-critic used by the worker
  agents/manager_agent.py      PPO actor-critic used by the manager
  training/train_worker.py     Worker PPO training
  training/train_manager.py    Manager PPO training
  evaluation/evaluate.py       Manager evaluation
  evaluation/evaluate_worker.py Worker-only evaluation
  evaluation/phase_c2.py       Full Phase C2 training/screening driver
  utils/scenarios.py           Shared scenario generation and replay
  utils/metrics.py             Metric helpers

tools/
  phase_c2_finalize.py         Rebuilds final aggregate tables from saved raw bundles
  plot_ppo_worker_manager_results.py  Creates PPO-only line graphs

docs/
  PHASE_C2_FINAL_RESULTS.md    Final full experiment report
  PPO_WORKER_MANAGER_PRESENTATION_RESULTS.md  PPO-only presentation summary
  METRIC_DEFINITIONS.md
  OBSERVATION_ACTION_SPEC.md
  PHYSICAL_ASSUMPTIONS.md

outputs/results/
  phase_c2_full/               Checkpoints used by the saved comparisons
  phase_c2_final/              Final 200-scenario aggregate evaluation files
  ppo_worker_manager_presentation/  PPO-only graphs and tables
```

## Environment Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH='src'
```

Run tests:

```powershell
python -m unittest discover -s tests -v
```

Compile-check the source:

```powershell
python -m compileall src tools
```

## Simulation Model

The default system is:

| Quantity | Value |
|---|---:|
| UAVs | 3 |
| Physical entities | 5 |
| Sensors | 15 |
| Sensors per entity | 3 |
| Episode length | 300 slots |
| Slot duration | 1.0 s in final experiments |
| Manager horizon | 5 worker slots |
| UAV placement grid | 4 x 4 |
| Sensor buffer | one packet, keep latest |
| Sensor uplink power | fixed maximum in the saved Phase C2 experiments |
| Backhaul power | selected by the manager PPO |
| Access bandwidth | fixed |
| Backhaul bandwidth | fixed, equally split over directed UAV-UAV links |
| Geometry | 2D horizontal model, no altitude |

Sensor packets arrive into one-packet buffers. If a new packet arrives while an older packet is pending, the old packet is overwritten. A worker action is allowed only when the selected sensor has a pending packet, the same sensor is not selected by two UAVs, and the predicted end-to-end delay can complete within the current slot.

## AoI and AoDT

Each sensor has an Age of Information value:

```math
\Delta_i(t)
```

Each entity has three sensors. Entity AoDT is the maximum sensor age inside the entity:

```math
\Delta_e(t) = \max_{i \in \mathcal{N}_e} \Delta_i(t)
```

This means the entity is considered fresh only if all required sensor streams are fresh. The reported AoDT is the average entity AoDT over entities, slots, episodes, and scenarios.

When a sensor is served successfully, its AoI is reset using the packet waiting time plus normalized end-to-end delay. If it is not served, its AoI increases by one slot.

## Fast Worker Controller

The worker acts every slot. For each UAV, it chooses either:

```text
one sensor to serve, or idle
```

The action constraints are:

- one sensor per UAV per slot;
- one UAV per sensor per slot;
- no service for sensors without pending packets;
- no service if the predicted uplink + backhaul + processing delay exceeds the slot duration;
- idle is always allowed.

The final worker uses fixed maximum uplink power. This is because there is no uplink-energy constraint in the final mathematical problem, so higher uplink power only decreases delay and has no physical trade-off in the implemented objective.

### Greedy Worker

The greedy worker evaluates feasible sensor-UAV pairs and selects the pair with the largest immediate predicted age/AoDT reduction, then repeats while preventing duplicate sensor selections. This is deterministic and uses the same feasibility rules and delay equations as the simulator.

### PPO Worker

The PPO worker was implemented as a masked actor-critic policy. It receives the worker observation, produces categorical logits over sensors plus idle for each UAV, applies feasibility masks, samples or deterministically selects actions, and uses PPO to update the actor and critic.

The worker reward used in the saved PPO-worker experiments is:

```math
r_t^w =
2.0 \frac{\bar{\Delta}(t)-\bar{\Delta}(t+1)}{10}
- \frac{\bar{\Delta}(t+1)}{10}
- 0.05\frac{N_{\mathrm{invalid}}}{M}
- 0.01\frac{N_{\mathrm{wasted}}}{M}
```

Interpretation:

- the first term rewards reducing average entity AoDT;
- the second term penalizes the current AoDT level;
- the invalid term penalizes infeasible attempted actions;
- the wasted term penalizes idle/wasted service opportunities;
- `M` is the number of UAVs.

This reward is useful for training a PPO scheduler, but it is also indirect: the policy receives one scalar reward after the AoDT update, so the exact sensor that improved an entity bottleneck is not always obvious from the reward alone. This is why I also kept a deterministic age-reduction worker as a comparison point.

## Slow Manager Controller

The manager acts every `H=5` worker slots. Its action contains:

1. one grid index for each UAV;
2. one feasible complete DT-host assignment for all entities;
3. one continuous backhaul power value per UAV.

The DT-host action uses feasible complete-assignment enumeration, so every executed host assignment satisfies:

```math
\sum_m x_{e,m}=1,
\qquad
\sum_e S_e x_{e,m} \le C_m.
```

The manager does not optimize access bandwidth, backhaul bandwidth, CPU frequency, packet arrivals, or channel parameters. Those are fixed simulation parameters.

## Backhaul Energy and Virtual Queues

Backhaul is needed when the serving UAV is not the DT-host UAV of the served sensor's entity. The backhaul transmission uses the source UAV's selected backhaul power.

For each UAV, the manager tracks a virtual queue:

```math
Z_m(k+1)=\max\left(0, Z_m(k)+\bar{E}^{bh}_m(k)-E^{max}_m\right)
```

where:

- `k` is the manager-window index;
- `\bar{E}^{bh}_m(k)` is average per-slot backhaul energy for UAV `m` in that manager window;
- `E^{max}_m` is the backhaul-energy budget.

The manager reward is Lyapunov-guided:

```math
r_k^m =
-\left[
V \frac{\bar{\Delta}_k}{\Delta_{\mathrm{norm}}}
+ \frac{1}{M}\sum_m
\frac{Z_m(k)}{E^{max}_m}
\frac{\bar{E}^{bh}_m(k)}{E^{max}_m}
\right]
```

The old queue `Z_m(k)` is used in the reward. The queue is then updated after the window. The implementation is Lyapunov-guided, not a formal stability-proof derivation.

## PPO Details

The PPO implementation is standard actor-critic PPO:

1. collect rollout transitions;
2. store observations, actions, rewards, log probabilities, values, and done flags;
3. compute discounted returns and GAE advantages;
4. recompute new action log probabilities;
5. form the PPO ratio;
6. apply clipped actor loss;
7. train the critic with value loss;
8. add entropy bonus for exploration;
9. update using mini-batches for several PPO epochs;
10. save checkpoints using validation performance.

Important PPO parameters:

| Parameter | Meaning |
|---|---|
| learning rate | size of neural-network weight updates |
| gamma | discount factor for future rewards |
| GAE lambda | smooths advantage estimates |
| clip epsilon | limits how far the new policy moves from the old policy |
| entropy coefficient | encourages exploration |
| value coefficient | weight of critic loss |
| rollout size | number of transitions collected before a PPO update |
| mini-batch size | number of rollout samples per gradient batch |
| PPO epochs | number of passes through the rollout |
| gradient clipping | prevents unstable large updates |

For the Phase C2 manager study, I screened manager rollout size, learning rate, and entropy coefficient using validation scenarios. The greedy-worker manager comparison used:

```text
rollout size = 128
learning rate = 1e-4
entropy coefficient = 0.01
manager seeds = 51, 52, 53, 54, 55
manager training = 1000 episodes per seed
validation scenarios = 40
```

For the PPO-worker hierarchy, two manager candidates were kept:

```text
Candidate 0: rollout 128, learning rate 1e-4, entropy 0.01
Candidate 1: rollout 128, learning rate 1e-4, entropy 0.005
```

The PPO worker was trained with:

```text
worker seeds = 41, 42, 43
worker training = 500 episodes per seed
worker validation scenarios = 40
worker power mode = fixed_max
```

## Final Results

Final evaluation used 200 held-out shared scenarios with deterministic learned policies. The same scenario suite was used for all policies.

| System | Mean AoDT | Mean backhaul energy | Positive violation | Feasible episodes |
|---|---:|---:|---:|---:|
| PPO worker + PPO manager C0 | 5.5322 | 0.0474 | 0.0000 | 0.955 |
| PPO worker + PPO manager C1 | 5.5185 | 0.0340 | 0.0000 | 1.000 |
| Greedy worker + PPO manager | 3.4951 | 0.0255 | 0.0000 | 1.000 |
| Static heuristic manager | 6.0915 | 0.1314 | 0.0030 | 0.650 |
| Random manager | 7.3638 | 0.2587 | 0.0148 | 0.000 |
| Fixed global manager | 8.2922 | 0.7019 | 0.4519 | 0.000 |

These results show the behavior of the PPO hierarchy and the comparison systems under the same 200 held-out scenarios. The PPO manager achieved low backhaul-energy usage and zero positive energy violation in both PPO-worker manager candidates and in the greedy-worker comparison.

## Saved Checkpoints

Greedy-worker PPO-manager checkpoints:

```text
outputs/results/phase_c2_full/manager_greedy_worker/models/manager_greedy_r128_lr1e-04_e0.01_seed51.pt
outputs/results/phase_c2_full/manager_greedy_worker/models/manager_greedy_r128_lr1e-04_e0.01_seed52.pt
outputs/results/phase_c2_full/manager_greedy_worker/models/manager_greedy_r128_lr1e-04_e0.01_seed53.pt
outputs/results/phase_c2_full/manager_greedy_worker/models/manager_greedy_r128_lr1e-04_e0.01_seed54.pt
outputs/results/phase_c2_full/manager_greedy_worker/models/manager_greedy_r128_lr1e-04_e0.01_seed55.pt
```

PPO-worker checkpoints:

```text
outputs/results/phase_c2_full/worker/models/worker_fixed_max_seed41.pt
outputs/results/phase_c2_full/worker/models/worker_fixed_max_seed42.pt
outputs/results/phase_c2_full/worker/models/worker_fixed_max_seed43.pt
```

PPO-worker manager Candidate 0 and Candidate 1 checkpoints are under:

```text
outputs/results/phase_c2_full/manager_final_candidate_0/models/
outputs/results/phase_c2_full/manager_final_candidate_1/models/
```

## Result Files and Graphs

Final aggregate results:

```text
outputs/results/phase_c2_final/
```

The multi-GB raw per-window/per-transition bundles are kept locally and ignored by Git so the GitHub repository remains shareable. The pushed repository keeps the compact summaries, selected checkpoints, presentation graphs, and documentation.

PPO-worker + PPO-manager presentation graphs:

```text
outputs/results/ppo_worker_manager_presentation/
  c1_aodt_improvement_over_time.png
  c1_backhaul_energy_over_time.png
  ppo_manager_validation_aodt.png
  ppo_manager_validation_energy.png
```

Detailed reports:

```text
docs/PHASE_C2_FINAL_RESULTS.md
docs/PPO_WORKER_MANAGER_PRESENTATION_RESULTS.md
```

## Useful Commands

Rebuild final aggregate CSV/JSON files from the saved raw bundles without reevaluating policies:

```powershell
$env:PYTHONPATH='src'
python tools/phase_c2_finalize.py --aggregate-only
```

Regenerate the PPO-only presentation graphs:

```powershell
$env:PYTHONPATH='src'
python tools/plot_ppo_worker_manager_results.py
```

Train a worker PPO model:

```powershell
$env:PYTHONPATH='src'
python -m dt_uav_v2.training.train_worker --episodes 500 --rollout-size 512 --randomize-scenarios --power-mode fixed_max --seed 41 --save-path outputs/results/phase_c2_full/worker/models/worker_fixed_max_seed41.pt
```

Train a PPO manager with the PPO worker using the current default manager PPO settings:

```powershell
$env:PYTHONPATH='src'
python -m dt_uav_v2.training.train_manager --episodes 1000 --rollout-size 128 --randomize-scenarios --worker-policy ppo --worker-model-path outputs/results/phase_c2_full/worker/models/worker_fixed_max_seed41.pt --save-path outputs/results/example_manager_ppo_seed51.pt --seed 51
```

Train a PPO manager with the greedy worker using the current default manager PPO settings:

```powershell
$env:PYTHONPATH='src'
python -m dt_uav_v2.training.train_manager --episodes 1000 --rollout-size 128 --randomize-scenarios --worker-policy greedy --save-path outputs/results/example_manager_greedy_seed51.pt --seed 51
```

The full Phase C2 experiment driver is preserved in:

```text
src/dt_uav_v2/evaluation/phase_c2.py
```

The saved outputs should be used for reporting unless a new experiment is intentionally started.

## Main Limitations

- The geometry is 2D and does not include UAV altitude.
- UAV movement and DT migration do not have propulsion or migration costs.
- The PPO manager often learned low-switching or static-like actions in the final evaluation.
- The PPO worker reward gives indirect credit for bottleneck sensor choices, which is an important point to discuss when interpreting the PPO-worker results.
- There is no classical mixed-integer optimization benchmark in this cleaned version.
- The double-PPO system and the greedy-worker + PPO-manager system should be described as separate evaluated systems, not as the same controller.
