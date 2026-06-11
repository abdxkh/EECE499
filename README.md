# DT-UAV AoDT Final Project

This repository contains the final two-timescale DT-UAV AoDT experiment:

- a fast worker PPO policy for per-slot sensor scheduling and continuous uplink power,
- a slow manager PPO policy for UAV grid placement, DT hosting, and continuous backhaul power,
- a Lyapunov-inspired virtual queue penalty for the per-UAV backhaul energy budget,
- final evaluation against random and fixed manager baselines.

The final kept checkpoints are:

```text
outputs/models/worker_continuous_final.pt
outputs/models/manager_backhaul_final.pt
```

Old intermediate checkpoints and outdated explanation files were removed to keep the project clean.

## Environment

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH='src'
```

## Final Evaluation

To run the final paper-style comparison:

```powershell
python -m dt_uav_v2.evaluation.evaluate --episodes 100 --randomize-scenarios
```

The output intentionally keeps only:

- reward
- AoDT
- energy

Current final 100-episode randomized result:

```text
----------------------------------------------
policy          reward        aodt      energy
----------------------------------------------
ppo            -0.2166      4.3303      0.1214
random        -11.0990      4.8708      0.3021
fixed         -26.2872      4.3511      0.2510
----------------------------------------------
```

## Train Worker

The worker controls the per-slot sensor selected by each UAV and the continuous uplink transmit power.
The worker observation also includes the current normalized manager-selected backhaul power per UAV.

```powershell
python -m dt_uav_v2.training.train_worker --episodes 100 --rollout-size 512 --randomize-scenarios
```

Saved checkpoint:

```text
outputs/models/worker_continuous_final.pt
```

## Train Manager

The manager uses the frozen worker and controls:

- one grid location per UAV,
- one DT host UAV per entity,
- one continuous backhaul transmit power per UAV.

```powershell
python -m dt_uav_v2.training.train_manager --episodes 100 --rollout-size 64 --randomize-scenarios
```

Saved checkpoint:

```text
outputs/models/manager_backhaul_final.pt
```

## Run Full Pipeline

This trains missing final checkpoints and then evaluates:

```powershell
python -m dt_uav_v2.run_final
```

## What Is Fixed

These are not optimized by the current final experiment:

- CPU rate and CPU cycles per bit,
- sensor positions after environment reset,
- packet-arrival probability model,
- access bandwidth,
- backhaul bandwidth,
- noise, pathloss, and channel parameters.

## What Is Optimized

The learned policies optimize the reward through:

- worker sensor scheduling,
- worker continuous uplink power,
- manager UAV placement on the grid,
- manager DT-host placement,
- manager continuous backhaul power.

The final metrics reported are reward, average AoDT, and average backhaul energy.

More presentation details are in:

```text
docs/FINAL_PRESENTATION_GUIDE.md
```
