# Experimental design

| Goal | Design | Runs | Notes |
|---|---|---|---|
| Percentiles, distributions | `lhs` or `sobol` | `n` | Sobol: `n` a power of two; LHS is fine for n < 64 |
| Screening of D inputs | `morris` | `r·(D+1)` (r trajectories, `n` = r) | four levels; rank by μ* and σ; cheap first pass at D ≥ 6 |
| Variance decomposition | `saltelli` | `n·(D+2)` (`n·(2D+2)` with `second_order`) | `n` power of two ≥ 128 for stable S1/ST; expensive |

Rules:
- Size by budget first: `estimate` prints the planned range; a 10 s proxy member allows
  hundreds of runs per hour on a shared node, a 40 s full-field member tens.
- Never resample failed rows: the journal keeps them as failures; Saltelli indices are computed on
  the complete rows and the failure count is part of the result.
- Fix `seed_root`; the design is reproducible and the manifest stores the design points.
- Keep reference sets when you increase `n`: a new `n` is a new study, do not append.
- Sobol indices need the model response to be finite everywhere in the box; check the failure
  class of any crash (numerical vs walltime) before widening ranges.
