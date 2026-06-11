# Final DT-UAV AoDT Setup

This is the clean final setup for presentation and paper experiments.

## What Is Optimized

The final system optimizes:

1. Worker sensor scheduling.
2. Worker continuous uplink transmit power.
3. Manager UAV grid placement.
4. Manager DT placement.
5. Manager continuous backhaul transmit power per source UAV.
6. Average AoDT.
7. Backhaul energy-budget violations through a Lyapunov-inspired virtual queue.

The final system keeps these fixed:

1. CPU model and CPU rate.
2. Sensor locations after each reset.
3. Packet-arrival process.
4. Access bandwidth.
5. Backhaul bandwidth split.

## Final Checkpoints

```text
outputs/models/worker_continuous_final.pt
outputs/models/manager_backhaul_final.pt
```

## Final Run Command

```powershell
cd C:\Users\AUB\Desktop\DT_UAV_AoDT_v2\DT_UAV_AoDT_v2
$env:PYTHONPATH='src'
python -m dt_uav_v2.evaluation.evaluate --episodes 100 --randomize-scenarios
```

## Final Result

```text
policy          reward        aodt      energy
ppo            -0.2166      4.3303      0.1214
random        -11.0990      4.8708      0.3021
fixed         -26.2872      4.3511      0.2510
```

## Simple Explanation

The worker acts every slot and chooses which sensor each UAV serves and the continuous uplink power. The worker also observes the manager's current normalized backhaul power per UAV, so it can condition scheduling decisions on how expensive or slow forwarding may be. The manager acts every 10 slots and chooses UAV grid positions, DT hosts, and continuous backhaul power per transmitting UAV. AoDT is computed as the maximum sensor AoI for each entity, then averaged across entities. Backhaul energy is controlled using a Lyapunov-inspired virtual queue.
