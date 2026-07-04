# Physical Assumptions

## Included in the Current Model

- fixed access bandwidth
- fixed backhaul bandwidth
- equal directed-link backhaul bandwidth split
- 2D UAV and sensor geometry
- pathloss-based communication rates
- finite UAV placement grid
- window-level DT-host reconfiguration
- per-UAV backhaul-energy accounting
- max-based entity AoDT

## Deliberately Omitted in Phase B

- propulsion energy
- continuous UAV trajectories
- relocation travel time
- DT migration delay
- DT migration energy
- backhaul interference coupling beyond equal bandwidth splitting
- uplink-energy constraint or objective

## Same-Step Completion Abstraction

When a sensor update is served, the code applies the AoI reset in the same decision step even if the computed end-to-end delay exceeds one slot.

This means:

- packet service is not carried across multiple discrete slots;
- delay enters freshness through `delay / slot_duration`;
- the model is a coarse slot-based freshness abstraction, not a packet-level transmission simulator.

## Quasi-Static Manager Decisions

Within one manager window:

- UAV grid positions remain fixed;
- DT hosts remain fixed;
- backhaul powers remain fixed.

These decisions are treated as quasi-static window-level configurations rather than explicit flight-control or migration-control trajectories.
