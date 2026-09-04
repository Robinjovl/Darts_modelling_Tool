# Well placement (`design.driver = "exhaustive"`)

Design keys: `well` (name to relocate), `objective` (only `cumulative_oil` is implemented; the
key is echoed, not switched on), `min_spacing_m`, `max_candidates` (seeded subset; omit for all
feasible cells). Parameters: one entry `{"name": "<well>", "family": "ScalarParam",
"args": {"target": "well_xyz"}}` naming the relocated well (the schema requires a parameter; the
driver takes the well from `design.well`). Observations give producers, quantities and report
times.

Procedure (`workflows/optimize.py`): candidates = all active cells at ≥ `min_spacing_m` from
every other well; baseline + each candidate simulated in isolated processes; `candidates.json`
holds every objective; `optimize_summary.json` the best and the baseline regret.

Cost: `1 + n_candidates` simulations (431 cells on the proxy → about 1.2 h serial at 10 s each;
`max_workers` parallelizes). Full-field beds (tens of thousands of cells) need a subset or a
surrogate — see `drivers.md`.

Rules:
- Same report grid and horizon for every candidate.
- Relocating a well keeps its controls; say so, or add a control study.
- Vertical wells only (the adapter overrides the well's x, y and keeps the perforation logic
  of the model).
- Report the best cell with its centroid and the top-5 spread; a flat top means the choice
  is not resolved at this budget.
