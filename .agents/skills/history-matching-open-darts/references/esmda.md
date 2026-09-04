# ES-MDA (`workflow: "hm-esmda"`)

Design keys: `ne` (ensemble size), `n_steps` (default 4), `alphas` (default equal
`n_steps` each, must sum of inverses = 1), `localization_length_m` (Gaspari–Cohn taper on
cell-to-well distance; 0 disables), `bounds_sigma` (clip updates to mean ± k·σ in log10),
`field` (the `LogPermField` parameter name).

Cost: `(n_steps + 1)·ne` simulations (posterior wave included); estimate before running.

Procedure implemented in `workflows/esmda.py`:
1. Prior: `ne` log10-perm fields from the declared family (geology stream).
2. For each step: forward ensemble (isolated processes), Kalman-like update with inflated
   noise `alpha_k·σ²`, localization on the P × nd taper, clipping to bounds.
3. Posterior forward wave; gates per step; `esmda/step_<k>_{params,data}.npy`.

Guidance:
- `ne = 50–100` with localization for full-cell fields; `ne = 6–10` only for smoke tests.
- `localization_length_m` about the well spacing; smaller on dense data.
- Do not tune the noise to pass chi²: change the parameterization or add inputs instead.
- Twin tests use the truth's seed index `TRUTH_SEED_INDEX = 999`, disjoint from the prior.
- Full (correlated) noise covariance is not supported; state this limitation when data errors
  are correlated (e.g. allocated rates).
