# Problem formulation

- Objective: `cumulative_oil` (implemented) = trapezoid of producer oil rate over report times
  (open-DARTS rates are m3/day, so cumulative volumes are m3; never relabel them as STB);
  NPV, net energy, CO2 stored and gas utilization follow the same pattern once an adapter
  exposes the quantity. State units and sign (maximize).
- Decision variables: `well_xyz` (placement by well name), BHP/rate controls, well index
  (completion), injected composition (simplex transform). Only `well_xyz` has a driver.
- Constraints: feasibility is decided before simulation (`min_spacing_m` to other wells, active
  cell); infeasible candidates are recorded, not simulated.
- Horizon and report times: the objective is only as accurate as the report grid; keep the
  grid at ≤ 1/10 of the horizon and identical across candidates.
- Baseline: the current configuration is always simulated as member `baseline`; regret uses it.
- Budget: `max_candidates` limits exhaustive runs (a seeded subset of the feasible set); state
  the fraction of the feasible set covered.
- Reporting: best decision, objective, `regret_of_baseline`, number feasible/infeasible/failed,
  simulations and wall time. "Uninformative" (regret `None`) means baseline and best coincide
  within tolerance: say so instead of claiming an improvement.
