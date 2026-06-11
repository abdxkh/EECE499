# Power Optimization Explanation

This file explains how the current code optimizes power and the other main decisions in the DT-UAV AoDT project.

## 1. Big Picture

- The project has two PPO agents:
  - **Worker PPO**:
    - Runs at the fast time scale.
    - Makes one decision every slot.
    - Chooses which sensor each UAV serves.
    - Chooses the continuous uplink transmit power for each UAV.
  - **Manager PPO**:
    - Runs at the slow time scale.
    - Makes one decision every manager window.
    - Chooses UAV grid positions.
    - Chooses which UAV hosts each entity Digital Twin.
    - Chooses continuous backhaul transmit power for each UAV.

- The current final experiment optimizes:
  - Sensor scheduling.
  - Worker uplink power.
  - UAV placement.
  - DT hosting.
  - Manager backhaul power.

- The current final experiment does **not** optimize:
  - CPU rate.
  - CPU cycles per bit.
  - Access bandwidth.
  - Backhaul bandwidth.
  - Sensor locations after reset.
  - Packet-arrival probability model.
  - Channel/pathloss/noise constants.

## 2. Important Config Values

- File:
  - `src/dt_uav_v2/config.py`

- Worker continuous uplink power:
  - Config key:
    - `worker_continuous_power`
  - Current value:
    - `True`
  - Meaning:
    - The worker does not choose from fixed power indexes only.
    - It outputs a continuous power value between the minimum and maximum sensor power levels.

- Backhaul power optimization:
  - Config key:
    - `optimize_backhaul_power`
  - Current value:
    - `True`
  - Meaning:
    - The manager chooses one continuous backhaul transmit power for each UAV.

- Backhaul power range:
  - Config keys:
    - `backhaul_power_min = 0.1`
    - `backhaul_power_max = 1.0`
  - Meaning:
    - Each UAV backhaul power is clipped to:

\[
0.1 \le p_m^{bh} \le 1.0
\]

- Default backhaul power:
  - Config key:
    - `backhaul_power = 1.0`
  - Meaning:
    - Used as the default before the manager sets optimized powers.

## 3. Worker Neural Network

- File:
  - `src/dt_uav_v2/agents/ppo.py`

- Class:
  - `ActorCritic`

- Function:
  - `forward`

- The worker network receives the worker observation:

\[
o_t^w
\]

- It passes the observation through two hidden layers:

\[
h_t^w = \tanh(W_2 \tanh(W_1 o_t^w + b_1) + b_2)
\]

- The worker has three outputs:
  - Sensor-selection logits.
  - Power-distribution parameters.
  - Critic value.

### 3.0 Worker Observation

- File:
  - `src/dt_uav_v2/envs/worker_env.py`

- Function:
  - `_state_to_obs`

- The worker observation contains:
  - pending packet flags,
  - packet waiting times,
  - packet sizes,
  - sensor AoI values,
  - entity AoDT values,
  - sensor-to-UAV distances,
  - DT-host one-hot encoding,
  - sensor-to-entity one-hot encoding,
  - each sensor's current DT-host one-hot encoding,
  - current manager-selected backhaul powers.

- The backhaul power part is:

\[
\tilde{p}_m^{bh}
=
\frac{p_m^{bh}}{p_{\max}^{bh}}
\]

- This gives the worker one value per UAV.

- Meaning:
  - The worker knows whether serving a sensor through UAV \(m\) requires backhaul.
  - The worker also knows the current backhaul power selected by the manager for UAV \(m\).
  - Therefore, the worker can learn that forwarding through a low-power backhaul UAV may be slower and should sometimes be deprioritized.

### 3.1 Sensor Logits

- Code:
  - `self.sensor_head = nn.Linear(hidden_dim, num_uavs * self.num_sensor_actions)`

- Meaning:
  - For each UAV, the network outputs scores for:
    - every real sensor,
    - plus one idle action.

- These scores are called **logits**.

- Logits are raw neural-network outputs before converting to probabilities.

- For UAV \(m\), the sensor probabilities are:

\[
\pi^w(s_m=i \mid o_t^w)
=
\frac{\exp(z_{m,i})}{\sum_j \exp(z_{m,j})}
\]

- The code uses a categorical distribution:
  - `Categorical(logits=sensor_logits)`

## 4. Worker Uplink Power Optimization

- File:
  - `src/dt_uav_v2/agents/ppo.py`

- Class:
  - `ActorCritic`

- Function:
  - `forward`

- Because `worker_continuous_power=True`, the worker power head is:

```python
self.power_head = nn.Linear(hidden_dim, num_uavs * 2)
```

- This means:
  - For each UAV, the network outputs two numbers.
  - These two numbers become the parameters of a Beta distribution.

- The code reshapes them:

```python
power_output = power_output.view(-1, self.num_uavs, 2)
```

- Then the code applies:

```python
power_output = F.softplus(power_output) + 1.0
```

- This guarantees the Beta parameters are positive:

\[
\alpha_m^w,\beta_m^w > 0
\]

- The worker samples a normalized power:

\[
z_m^{ul} \sim \mathrm{Beta}(\alpha_m^w,\beta_m^w)
\]

- The normalized power is in:

\[
0 \le z_m^{ul} \le 1
\]

- File:
  - `src/dt_uav_v2/agents/ppo.py`

- Function:
  - `_format_env_action`

- The normalized action is converted to physical uplink power:

\[
p_m^{ul}
=
p_{\min}^{ul}
+
z_m^{ul}
\left(
p_{\max}^{ul} - p_{\min}^{ul}
\right)
\]

- In code:

```python
power_action = self.power_min + power_unit * (self.power_max - self.power_min)
```

- The environment receives actions like:

```python
(sensor_id, power_action)
```

- So for each UAV, the worker sends:
  - which sensor to serve,
  - what uplink transmit power to use.

## 5. How Worker Uplink Power Affects the Environment

- File:
  - `src/dt_uav_v2/envs/base_env.py`

- Function:
  - `step_worker`

- The worker-selected uplink power is clipped:

```python
power = float(
    np.clip(
        power_index,
        min(self.sensor_power_levels),
        max(self.sensor_power_levels),
    )
)
```

- The uplink rate is then calculated using:

```python
R_ul = self.uplink_rate(sensor_id, m, power)
```

- The uplink delay is:

\[
\tau_i^{ul}(t)
=
\frac{L_i(t)}{R_{i,m}^{ul}(t)}
\]

- Higher uplink power usually gives:
  - higher SNR,
  - higher uplink rate,
  - lower uplink delay,
  - lower total update delay,
  - potentially lower AoI/AoDT.

- Important:
  - The current worker reward does **not** include explicit sensor energy cost.
  - Therefore, worker uplink power is optimized mainly through its effect on delay and AoDT.

## 6. Worker Reward

- File:
  - `src/dt_uav_v2/envs/worker_env.py`

- Function:
  - `_compute_reward`

- The worker reward is:

\[
r_t^w
=
\text{AoDT improvement bonus}
-
\text{AoDT cost}
-
\text{invalid action cost}
-
\text{wasted slot cost}
\]

- In code:

```python
reward = aodt_delta_bonus - aodt_cost - invalid_cost - wasted_cost
```

- AoDT cost:

\[
\text{AoDT cost}
=
\frac{\overline{\Delta}(t)}{\text{aodt reward scale}}
\]

- AoDT improvement bonus:

\[
\text{bonus}
=
w_{\Delta}
\frac{
\overline{\Delta}(t-1)-\overline{\Delta}(t)
}{
\text{aodt reward scale}
}
\]

- Invalid action cost:

\[
\text{invalid cost}
=
c_{\text{invalid}}
\frac{
\text{invalid count}
}{
M
}
\]

- Wasted slot cost:

\[
\text{wasted cost}
=
c_{\text{wasted}}
\frac{
\text{wasted count}
}{
M
}
\]

- This means the worker learns to:
  - serve useful pending sensors,
  - avoid invalid actions,
  - avoid wasting UAV slots,
  - reduce AoDT.

## 7. Manager Neural Network

- File:
  - `src/dt_uav_v2/agents/manager_agent.py`

- Class:
  - `ManagerActorCritic`

- Function:
  - `forward`

- The manager receives the manager observation:

\[
o_n^m
\]

- It passes the observation through two hidden layers:

\[
h_n^m = \tanh(W_2 \tanh(W_1 o_n^m + b_1) + b_2)
\]

- The manager has four outputs:
  - UAV grid-placement logits.
  - DT-host logits.
  - Backhaul-power Beta parameters.
  - Critic value.

## 8. Manager UAV Placement Optimization

- File:
  - `src/dt_uav_v2/agents/manager_agent.py`

- Code:

```python
self.grid_head = nn.Linear(hidden_dim, num_uavs * num_grid_points)
```

- Meaning:
  - For each UAV, the manager outputs logits over all grid points.

- For UAV \(m\), the manager samples:

\[
g_m \sim \mathrm{Categorical}(\mathrm{softmax}(z_m^g))
\]

- The selected grid index is converted into an actual 2D UAV position.

- File:
  - `src/dt_uav_v2/envs/manager_env.py`

- Function:
  - `_apply_manager_action`

- Code:

```python
self.base_env.uav_positions = self.grid_points[uav_grid_indices].copy()
```

- This means the manager learns where to place each UAV on the grid.

## 9. Manager DT-Host Optimization

- File:
  - `src/dt_uav_v2/agents/manager_agent.py`

- Code:

```python
self.host_head = nn.Linear(hidden_dim, num_entities * num_uavs)
```

- Meaning:
  - For each entity, the manager outputs logits over UAVs.
  - The selected UAV becomes the Digital Twin host for that entity.

- For entity \(e\):

\[
x_e \sim \mathrm{Categorical}(\mathrm{softmax}(z_e^x))
\]

- File:
  - `src/dt_uav_v2/envs/manager_env.py`

- Function:
  - `_apply_manager_action`

- Code:

```python
self.base_env.dt_hosts = self.base_env._repair_dt_storage(dt_hosts.copy())
```

- Important:
  - If the selected DT hosts violate storage capacity, the environment repairs the assignment.
  - So the manager proposes DT hosts, then the simulator enforces storage feasibility.

## 10. Manager Backhaul Power Optimization

- File:
  - `src/dt_uav_v2/agents/manager_agent.py`

- Class:
  - `ManagerActorCritic`

- Because `optimize_backhaul_power=True`, the manager has:

```python
self.backhaul_power_head = nn.Linear(hidden_dim, num_uavs * 2)
```

- This gives two outputs per UAV.

- These two outputs become Beta-distribution parameters:

\[
\alpha_m^{bh},\beta_m^{bh}
\]

- Code:

```python
backhaul_power_output = self.backhaul_power_head(features)
backhaul_power_output = backhaul_power_output.view(-1, self.num_uavs, 2)
backhaul_power_output = F.softplus(backhaul_power_output) + 1.0
```

- The manager samples normalized backhaul power:

\[
z_m^{bh}
\sim
\mathrm{Beta}
\left(
\alpha_m^{bh},\beta_m^{bh}
\right)
\]

- This value is in:

\[
0 \le z_m^{bh} \le 1
\]

- File:
  - `src/dt_uav_v2/agents/manager_agent.py`

- Function:
  - `select_action`

- The normalized value is converted to real backhaul power:

\[
p_m^{bh}
=
p_{\min}^{bh}
+
z_m^{bh}
\left(
p_{\max}^{bh}-p_{\min}^{bh}
\right)
\]

- In code:

```python
env_action["backhaul_powers"] = (
    self.backhaul_power_min
    + power_unit * (self.backhaul_power_max - self.backhaul_power_min)
).astype(float)
```

- So the manager action contains:

```python
{
    "uav_grid_indices": ...,
    "dt_hosts": ...,
    "backhaul_powers": ...
}
```

## 11. How Manager Backhaul Power Is Applied

- File:
  - `src/dt_uav_v2/envs/manager_env.py`

- Function:
  - `_apply_manager_action`

- The manager-selected powers are clipped:

```python
self.base_env.backhaul_powers = np.clip(
    backhaul_powers,
    self.backhaul_power_min,
    self.backhaul_power_max,
).astype(float)
```

- This guarantees:

\[
p_{\min}^{bh}
\le
p_m^{bh}
\le
p_{\max}^{bh}
\]

- The selected powers are stored in:

```python
self.base_env.backhaul_powers
```

- Shape:

\[
(M,)
\]

- Meaning:
  - One backhaul transmit power per UAV.

## 12. Backhaul Rate Equation

- File:
  - `src/dt_uav_v2/envs/base_env.py`

- Function:
  - `backhaul_rate`

- If UAV \(m\) sends an update to DT-host UAV \(k\), the code computes:

\[
R_{m,k}^{bh}
=
B_{link}^{bh}
\log_2
\left(
1+
\frac{
p_m^{bh} h_{m,k}
}{
\sigma^2
}
\right)
\]

- The bandwidth per directed backhaul link is:

\[
B_{link}^{bh}
=
\frac{B^{bh}}{M(M-1)}
\]

- Code:

```python
num_links = self.M * (self.M - 1)
B_link = self.B_backhaul / num_links
snr = (power * h) / self.noise_power
rate = B_link * np.log2(1.0 + snr)
```

- If source UAV and DT-host UAV are the same:

\[
m=k
\]

- Then no backhaul is needed.

- Code returns a very large rate:

```python
return 1e18
```

## 13. Backhaul Delay and Energy

- File:
  - `src/dt_uav_v2/envs/base_env.py`

- Function:
  - `step_worker`

- Backhaul is required only when:

\[
m \ne \text{DT host of the sensor entity}
\]

- Code:

```python
if m != dt_host:
```

- Backhaul delay:

\[
\tau_{m,k}^{bh}(t)
=
\frac{L_i(t)}{R_{m,k}^{bh}(t)}
\]

- Code:

```python
tau_bh = packet_size / R_bh
```

- Backhaul energy:

\[
E_m^{bh}(t)
=
p_m^{bh}(t)\tau_{m,k}^{bh}(t)
\]

- Code:

```python
e_bh = bh_power * tau_bh
backhaul_energy[m] += e_bh
```

- Energy is charged to:
  - the source/forwarding UAV \(m\),
  - not the receiving DT-host UAV.

## 14. Main Backhaul Power Tradeoff

- If manager chooses higher backhaul power:
  - SNR increases.
  - Backhaul rate increases.
  - Backhaul delay decreases.
  - Total update delay may decrease.
  - AoDT may improve.

- But:
  - Backhaul energy is \(p_m^{bh}\tau_{m,k}^{bh}\).
  - Higher power can increase energy.
  - If energy exceeds the budget, the virtual queue grows.
  - The manager reward becomes worse.

- Therefore, the manager learns a balance:
  - use enough power to reduce delay,
  - but not too much power because of the energy queue penalty.

## 15. Manager Reward

- File:
  - `src/dt_uav_v2/envs/manager_env.py`

- Function:
  - `_compute_reward`

- The reward is:

\[
r_n^m
=
-
\left(
\frac{
\overline{\Delta}_n
}{
\Delta_{\text{norm}}
}
+
\beta
\frac{
\sum_m Q_m(n)[E_m(n)-E_{\max}]^+
}{
E_{\max}
}
\right)
\]

- Code:

```python
aodt_cost = avg_window_aodt / max(self.aoi_obs_norm, 1e-9)
queue_cost = self.lyapunov_beta * float(
    np.sum(self.virtual_queues * positive_violation)
) / max(self.energy_budget, 1e-9)

return -float(aodt_cost + queue_cost)
```

- Meaning:
  - Lower AoDT gives better reward.
  - Lower energy violation gives better reward.
  - Large virtual queues make future violations more expensive.

## 16. Virtual Queue Update

- File:
  - `src/dt_uav_v2/envs/manager_env.py`

- Function:
  - `step`

- At the end of one manager window, the environment computes average energy per UAV:

\[
E_m(n)
\]

- The violation is:

\[
v_m(n)
=
E_m(n)-E_{\max}
\]

- Positive violation is:

\[
[v_m(n)]^+ = \max(v_m(n),0)
\]

- The virtual queue is updated as:

\[
Q_m(n+1)
=
\max
\left(
0,
Q_m(n)+v_m(n)
\right)
\]

- Code:

```python
energy_violation = avg_energy_per_uav - self.energy_budget
positive_violation = np.maximum(energy_violation, 0.0)
self.virtual_queues = np.maximum(
    0.0,
    self.virtual_queues + energy_violation,
).astype(np.float32)
```

- Important:
  - If energy is above budget, the queue increases.
  - If energy is below budget, the queue decreases.
  - The queue cannot go below zero.

## 17. How PPO Optimizes the Decisions

- PPO does not solve the equations directly.

- PPO learns the neural-network weights:

\[
\theta
\]

- These weights determine:
  - sensor logits,
  - grid logits,
  - DT-host logits,
  - Beta parameters for continuous powers,
  - value estimates.

- During training:
  - The agent observes the state.
  - The neural network outputs action distributions.
  - The agent samples actions.
  - The simulator calculates AoDT, delay, energy, reward.
  - PPO updates the neural-network weights.

- The PPO probability ratio is:

\[
\rho_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
\]

- The clipped PPO objective uses:

\[
\min
\left(
\rho_t(\theta)A_t,
\mathrm{clip}
\left(
\rho_t(\theta),
1-\epsilon,
1+\epsilon
\right)A_t
\right)
\]

- File:
  - `src/dt_uav_v2/agents/ppo.py`
  - `src/dt_uav_v2/agents/manager_agent.py`

- Code:

```python
ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
unclipped = ratio * advantages[batch_idx]
clipped = torch.clamp(
    ratio,
    1.0 - self.clip_eps,
    1.0 + self.clip_eps,
) * advantages[batch_idx]

actor_loss = -torch.min(unclipped, clipped).mean()
critic_loss = (returns[batch_idx] - values).pow(2).mean()
entropy_loss = entropy.mean()
loss = actor_loss + self.value_coef * critic_loss
loss = loss - self.entropy_coef * entropy_loss
```

- PPO improves the policy by:
  - increasing probability of actions that gave good advantage,
  - decreasing probability of actions that gave bad advantage,
  - clipping the update so the policy does not change too suddenly,
  - training the critic to predict returns,
  - keeping some entropy so the agent explores.

## 18. Deterministic vs Stochastic Power Selection

- During training:
  - The agent samples power from the Beta distribution.
  - This gives exploration.

- During deterministic evaluation:
  - The code uses the mean of the Beta distribution:

\[
z_m
=
\frac{\alpha_m}{\alpha_m+\beta_m}
\]

- This makes evaluation stable and repeatable.

## 19. How Each Optimized Quantity Is Learned

- Sensor scheduling:
  - Learned by the worker sensor logits.
  - Better schedules reduce AoDT and avoid invalid/wasted actions.

- Uplink power:
  - Learned by the worker Beta power head.
  - Better uplink power can reduce uplink delay and improve AoDT.
  - There is no explicit uplink energy penalty in the current reward.

- UAV placement:
  - Learned by the manager grid logits.
  - Better UAV positions improve channel distances and reduce transmission delays.

- DT hosting:
  - Learned by the manager host logits.
  - Better DT hosting can reduce backhaul need.
  - If the serving UAV also hosts the DT, no backhaul is required.

- Backhaul power:
  - Learned by the manager Beta backhaul-power head.
  - Better backhaul power balances delay reduction against energy-budget pressure.

## 20. Presentation Summary

- The worker optimizes fast slot-level decisions:
  - which sensor to serve,
  - what uplink power to use.

- The manager optimizes slow window-level decisions:
  - where to place UAVs,
  - where to host Digital Twins,
  - what backhaul power each UAV should use.

- Power is continuous because the neural network outputs Beta-distribution parameters.

- The Beta sample is normalized in \([0,1]\).

- The normalized sample is scaled to the real power range.

- Backhaul power affects:
  - backhaul SNR,
  - backhaul rate,
  - backhaul delay,
  - backhaul energy,
  - AoDT,
  - virtual-queue penalty.

- PPO optimizes power by learning which power distributions produce better long-term reward.
