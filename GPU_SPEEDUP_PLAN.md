# open-DARTS GPU speedup plan (SPE10 / AMGX-CPR evidence base)

**Repo state:** `xiaoming/add-mgr` @ `5b8bccf9f` (post NonlinearSolverSpec refactoring, AMGX pin for CUDA 13.3, dirs `solvers/` → `linear_solvers/`).
**Evidence base:** SPE10 60×220×85 (1.122M cells, 2 vars), `AMGXCPRSolverSpec`, A100 80GB PCIe.
Profiled at 200 days (12.0s run) and **3650 days (38.1s sim — the calibration case for this plan)**:
32 timesteps, 137 Newtons, 1309 GMRES iters (~1446 preconditioner applies), 180 assemblies; reproducible ±0.2s.
All numbers below were adversarially re-verified by independent audits against nsys/ncu raw data and code
(profiling artifacts: `darts-models/shared/spe10_solver_bench/results/{long3650_base,nsys_long3650}`,
report `.../results/GPU_PERF_ANALYSIS_2026-07-12.md`; ncu reports in session scratchpad `ncu/`).

## 1. Where the 38.1s goes (3650-day sim timer, corrected decomposition)

| bucket | s | % | dominant content |
|---|---|---|---|
| linear solve | 12.34 | 32% | **ILU0 bsrsv2 trisolves 8.01** (1446×5.5ms), AMGX V-cycle apply 2.23 (172k tiny launches), SpMV 0.51, BLAS/glue ~1.6 |
| linear setup | 6.84 | 18% | **AMGX hierarchy rebuilt every Newton 5.01**, ILU0 refactorization 1.69 |
| host↔device | 6.02 | 16% | **180× 260.3MB `op_vals_arr` DtoH ≈ 5.0s** (pageable 8.9GB/s; 54.3GB total DtoH), RHS/dX/X 18MB ping-pong |
| jacobian assembly | 2.59 | 7% | host-side OBL point generation ~1.36 (serial, pybind per point), assembly kernel 0.47, host wells 0.48 |
| newton update (timed) | 2.05 | 5% | host `apply_newton_update`: composition correction (0.59, child node), chop, OBL-axis clamp, X update |
| **untimed residual** | **7.68** | 20% | host residual norms ~3.4 (serial 1.122M sweep/Newton) + well-residual `average_operator` second sweep 0.7–1.4 + RHS re-upload/mirror pieces ~1.0 + `post_newtonloop` FIPS/rates/Xn 0.3–0.5 + timer `cudaDeviceSynchronize` ~0.4 + **~1.5–2 unattributed Python/pybind loop (needs py-spy)** |

GPU has **no kernel running 56%** of the run; kernel busy total 17.7s (~300k launches, 7.5k/s).

Hardware-counter verdicts (ncu, identical at 200d and 3650d steady state — kernel character is dt-independent):
- **ILU0 trisolve: latency-bound, overdetermined.** `sm__throughput`==`issue_active`==40%, occupancy 90.8% of a 100% max, sectors/req 1.0 (warp-broadcast, 30.9× flop redundancy), spin-polls sync through L2 (~25–30 round-trips × 363 levels), DRAM 4.3%. **fp32 / occupancy / coalescing levers are dead (fp32 ceiling 2–4%).**
- **Assembly kernel: bound by L2 tag-request rate (79%)**, well-coalesced (7.15 sec/req), 71 regs intrinsic; only `__launch_bounds__`≤64 (+~0.02s) and the dead-diffusion guard (below) help.
- AMG coarse-level SpMVs run at 0.6–5 waves/SM — launch-bound by design; `cpr_setup_kernel` is the worst-coalesced owned kernel (20–24 sec/req) but only ~0.1s here.

## 2. The plan

Ordering = value/risk. Savings are strict-propagation ranges for the 3650d case; the % carries to any
Newton-dominated GPU run of this class. Dependencies: item 1 must land before/with 3c; item 2 is a mandatory
companion of item 1 (host loses ACC values).

### P0 — eliminate pure waste (low risk, no convergence change) → sim ≈ 27–28s
1. **Device ACC-op pack instead of the full `op_vals_arr` DtoH** — save **4.8–5.0s**.
   `engines/src/engine_base_gpu.cpp:147` copies all 29 ops × 1.122M × 8B after every assembly; host consumers
   need only the 2 ACC ops per Newton (`engine_base.cpp:1768/1725`) **plus perforation-row FLUX ops on the 32
   converged steps** (`ms_well::calc_rates`) — pack kernel + 18MB DtoH + ~100KB well-row gather.
   Effort ~1 day. Validation: residuals bit-identical; well rates identical.
2. **Device-resident residual norms — BOTH sweeps** — save **3.4–4.4s**.
   `calc_newton_residual_L2` (`engine_base.cpp:1768`) *and* `calc_well_residual`/`average_operator` (`:1725`,
   called at `:1858`) are each serial 1.122M-cell host sweeps per Newton. cub reductions (PV-weighted L2, op
   averages). Risk: FP reduction order can flip borderline Newton decisions — gate with NI/LI parity run.
   Effort 1–2 days.
3. **Mechanical fixes** — save **~1.3s**: (a) guard the unconditional 18MB RHS re-upload
   (`darts/models/darts_model.py:1546`, no-op check already exists in `apply_rhs_flux`) ~0.3; (b) stage only
   well-head RHS rows instead of full 18MB to `RHS_wells_d` (`engine_super_gpu.tpp:977`) ~0.3; (c) skip
   `op_vals_arr_n = op_vals_arr` 260MB host mirror + move FIPS/calc_rates consumption to device data or
   once-per-step (`engine_base.cpp:3338-3359`; GPU engine already keeps device mirrors,
   `engine_base_gpu.cpp:116`) ~0.7; (d) skip AMGX x-upload (guess is zeroed) ~0.05. Hours each.
4. **`timer_node` clock-sync opt-in** — save **~0.4s** and get honest profiles: every timer start/stop calls
   `cudaDeviceSynchronize` in GPU builds (~55% of all syncs). Hours.
5. **Pin the remaining host buffers** (`cudaHostRegister` X/dX/RHS + pack buffer; zero pinned memory exists in
   engines today, `gpu_tools.h:83`) — save **0.35–0.5s** after items 1/3 shrink the pool. Hours.

### P1 — setup amortization + interpolator (low-medium risk) → sim ≈ 23–24s
6. **Adaptive AMG hierarchy reuse in `linsolv_amgx`** — save **2.0–3.0s** of the 5.01s setup.
   Mechanism verified in vendored AMGX (`structure_reuse_levels` honored from 2nd setup, `amg_solver.cu:188`;
   `AMGX_solver_resetup` exported). ⚠ **Blind freeze is experimentally fatal** (json `structure_reuse_levels:-1`
   probe: preconditioner collapse → Newton divergence → unbounded OBL point generation → hang). Port the CPU
   BCSR-CPR `adaptive_li` policy: reuse while LI stays within threshold, force rebuild on growth. ~1 day.
7. **Batch + parallelize adaptive-OBL point generation** — save **0.7–1.0s**. The GPU interpolator generates
   missing points one virtual/pybind `evaluate()` at a time on host (~1.36s of the 1.53s "gpu interpolation"
   timer; the GPU kernels are only 0.17s). Mirror the CPU `materialize_missing_cache` batching + OpenMP.
   0.5–1 day.
8. **Dead-diffusion guard in the assembly kernel** — save **0.15–0.2s**. The diffusion block
   (`engine_super_gpu.tpp:595-678`) executes unconditionally; for models without diffusion it is ~40% of the
   loop's load requests against the measured L2-tag bound. Guard on `tranD != 0` / engine flag. Hours.
9. **py-spy/nvtx characterization of the ~1.5–2s unattributed Python per-Newton residue** (diagnostic gate for
   P2 scope; no saving claimed).

### P2 — solver algorithm + host Newton (medium risk, needs iteration studies) → sim ≈ 16–18s
10. **Replace exact ILU0 stage-2 with AMGX multicolor DILU (or Chebyshev)** — net **4–5.5s**.
    Pool = 8.01 trisolve + 1.69 factor. New stage-2 apply ~1.1–1.6ms → 2.5–3.0s at LI+30%, **minus ~1.3s
    LI-growth tax** on AMGX-apply/SpMV/BLAS. Requires an LI study across dt ramp (SPE10 heterogeneity is the
    risk); pair with `strength_threshold 0.5` **only if measured net-positive after reuse** (its −21% LI and
    +27% setup both shrink once 6/10 land). If LI growth ≤15%, net rises to ~5.5–6s. 0.5–1 day + study.
11. **Device-resident Newton update** — save **1.8–2.0s** (+0.2s ping-pong not double-counted with 5).
    Port `apply_newton_update` (`engine_base.cpp:2061-2135`): composition correction, global/local chop,
    OBL-axis clamp, X update — 4+ kernels, must reproduce host semantics exactly. 2–4 days.
12. **Wells on device / trimmed** — save **0.3–0.4s**: host well assembly 0.48 (`engine_super_gpu.tpp:968`),
    per-Newton `check_constraints` (`engine_base_gpu.h:127`), well-controls interp 0.21 (or build well itors
    with `platform='cpu'`). 1–2 days.
13. **Launch/sync campaign** — save **0.3–0.7s**: AMGX config surgery against the 172k coarse-level launches
    (aggressive coarsening / fewer levels / larger `min_coarse_rows` / coarse Jacobi — cheap json experiments,
    ceiling ~0.5–1s), batch GMRES CGS2 scalar readbacks, drop the 2 redundant CPR deviceSyncs, fuse the 348×
    18MB `cudaMemset` into `cpr_solve_prolongate`. CUDA graphs remain blocked for `AMGX_solver_solve`.
14. **Stream overlap: ILU0 refactorization ∥ AMGX setup** (independent inputs) — up to ~1s while setup is
    still rebuilt every Newton; partially voided after item 6 lands. Optional, evaluate after 6.

### Explicitly rejected (measured/verified dead ends — do not spend time here)
- **fp32 ILU0 trisolve**: latency-bound; ceiling 2–4% (~0.2s here). The unwired `gpu_gmres_cpr_amgx_ilu_sp`
  enum also silently falls back when `ENABLE_BOS_SOLVERS=OFF` (`engine_base_gpu.h:482`) — fix the silent
  fallback as hygiene, not as a perf item.
- **Occupancy/coalescing tuning of trisolve or assembly**: trisolve occupancy is at its theoretical max;
  assembly is L2-tag-bound and well-coalesced (block-size changes are a provable no-op).
- **Blind AMG hierarchy freeze** (`structure_reuse_levels` without gating): reproducibly hangs SPE10.
- **`interpolator D1`**: LI ×10 (measured 70.8s vs 12.0s at 200d).
- **Stage-1 engine swap** (hypre-GPU / own AMG): effort-disproportionate to a 2.23s apply pool.
- **Multi-GPU at ~1M cells**: trisolve is latency-bound and coarse AMG launch-bound — splitting worsens both.
  Revisit ≥10M cells.
- **Assembly-over-previous-solve overlap**: impossible (dX → update → assembly dependency).

### Expected trajectory (3650-day case, strict propagation)

| stage | sim time | speedup |
|---|---|---|
| today | 38.1s | 1.0× |
| after P0 | 27–28s | ~1.4× |
| after P1 | 23–24s | ~1.6× |
| after P2 | **16–18s** | **~2.1–2.4×** |

Floor analysis: post-P2 kernel busy ≈ 9–11s (stage-2 apply, AMG setup+apply with LI tax, SpMV/BLAS, assembly)
+ irreducible host/Python ~3–4s → **~15–18s is the realistic floor of this ladder**. Going below needs the
AMG-apply launch campaign at the AMGX-internals level plus the Python-residue work from item 9.
The same items cut the 200-day run 12.0 → ~5.5–7s; wall_total additionally carries ~4.3s init
(imports 1.6, engine init 1.5 — includes the one-time 0.85s cuSPARSE trisolve analysis — reservoir 1.2),
one-time and out of scope here.

## 3. Validation protocol (every item)
1. `run_bench.py --config mr_amgx_cpr` at 200d AND 3650d; gates: identical timesteps/NI/LI (or justified
   delta for solver-algorithm items), well rates match reference, timer tree improvement matches the claim.
2. Items touching residuals/updates (2, 10, 11): parity run with tightened `tol_newton` to confirm no
   trajectory drift; watch the fail-open risk — a degraded preconditioner must cut the timestep, not spiral
   into unbounded OBL point generation (add a point-generation budget guard; same family as WI#105).
3. Keep `results/long3650_base` as the pinned baseline; one variance repeat per measurement.

## 4. Side findings to file separately
- **Suspected thermal-only Jacobian bug**: `engine_super_gpu.tpp:690` inner loop shadows the thread's column
  index `v` — energy-row columns each receive the full row-sum; verify against `engine_super_cpu.tpp` and fix
  (does not affect isothermal runs).
- Benchmark script `run_bench.py` now carries compat shims for the API rename
  (`set_sim_params(tol_linear=)` → spec `tolerance`, `m.solver` → `m.linear_solver`,
  `darts.solvers` → `darts.linear_solvers`) — other model scripts will need the same migration.
- AMGX config is loaded from `amgx_config_bs<N>.json` in the **current working directory** only
  (`linear_solvers/src/linsolv_amgx.cpp:56`) — add an env-var override + resolved-path log line.

---

# Implementation status — 2026-07-18 (staged, not committed)

Implemented and verified on SPE10 `mr_amgx_cpr` (A100, `solvers` env). 16 files, +390/−44.

| run | before | after P0 | after P1 | check |
|---|---|---|---|---|
| 200d run_s | 12.17 | 7.64 | **7.3–7.4** | P0: NI/LI bit-identical (43/305); P1: 44/345 |
| 3650d sim_s | 38.1 | 26.1 | **22.2 (−42%)** | P0: bit-identical (137/1309); P1: NI 133, LI 1452 (+11%) |
| 3650d wall_total | 42.8 | — | **27.0** | |

**P0 (parity-exact, all landed):**
1. `op_vals_arr` 260MB-per-assembly DtoH removed; L2 residual norms + `average_operator` reduce on
   device (`engine_base_gpu.cpp` kernels); host mirror refreshed only on accepted timesteps
   (`post_newtonloop` predicate); L1/Linf fall back via `sync_op_vals_to_host()`.
2. Host `op_vals_arr_n` mirror skipped on GPU (`keep_host_op_vals_n_mirror`).
3. RHS re-upload guarded in `darts_model.py` (only when `set_rhs_flux` overridden / DFM wells).
4. Well-head RHS rows copied directly into `RHS_d`; 18MB scratch `RHS_wells_d` deleted (4 engines).
5. AMGX `x`-upload → `AMGX_vector_set_zero` (sizing without transfer).
6. timer `cudaDeviceSynchronize` opt-in via `DARTS_TIMER_SYNC=1` (default off; sub-node GPU
   attribution becomes async-skewed unless set).
7. `cudaHostRegister` pinning of X/dX/RHS/op_vals_arr.

**P1 (landed):**
8. Dead-diffusion guard `tranD != 0` in `engine_super_gpu.tpp` assembly loop (exact-zero skip).
9. **Adaptive AMG hierarchy reuse** in `linsolv_amgx`: `structure_reuse_levels: 999` injected into
   the default config; reuse gated by outer-LI feedback (`set_last_outer_iters` chain
   gmres_gpu→cpr→amgx), rebuild after `DARTS_AMGX_REUSE` (default 2) consecutive reuses, on LI
   degradation > `DARTS_AMGX_REUSE_GROWTH` (default 1.5×) vs fresh-hierarchy baseline, and after any
   solve failure. `DARTS_AMGX_REUSE=0` restores the old behavior exactly.
   ⚠ empirically `-1` ("reuse all") is broken in this AMGX build — even the FIRST hierarchy build
   under it produces a diverging preconditioner; the positive level count works.
   3650d: setup 6.84→3.80s, NI 137→133, LI +11%.

**Discovered en route:** the OBL batch pre-pass (plan item 7) already exists in
`multilinear_adaptive_gpu_interpolator.tpp` (single `evaluate_batch` over unique missing points) —
the remaining ~1.5s is per-call overhead + serial in-batch evaluation (OMP needs an evaluator
thread-safety audit).

**P2 status:** stage-2 swap hook landed (`DARTS_CPR_STAGE2=amgx` + `amgx_config_bs2.json`), but the
MULTICOLOR_DILU experiment hangs before the first timestep (AMGX bs2 wrapper integration —
coloring/vector semantics) → needs its allocated follow-up study. Device Newton update, wells,
launch campaign: not started (next round per plan).

## Branch-consistency notes — 2026-07-18 (commits 18c0c4f5a, 01771016a, 147b564bc)

- **`origin/nonlinear_refactoring`** (merges before this branch; `cbe6327cb` re-enables OBL bounds as
  part of update via `OBLBoundsSpec(mode='obl_axes', axis_min/max)` → `engine.correct_obl_axes(min,max)`,
  with the convergence verdict moved to the Python `NonlinearSolver`, incl. NaN-residual rejection):
  the GPU op_vals refresh no longer duplicates the C++ convergence predicate — it rides a virtual
  `sync_host_data_for_accepted_step()` hook called from inside the accepted-step branch itself, so it
  follows the verdict wherever that logic lives after the merge. The granular `correct_*` pipeline
  operates on host X/dX; this branch deliberately keeps the dX-DtoH / X-HtoD hops (plan item P2-11
  must be coordinated with that pipeline post-merge: port the `correct_*` hooks to device versions).
- **`origin/development`**: already fully merged here (`8483d4e18`); the "parallel batch evaluation
  for multilinear_adaptive_gpu_interpolator" (`8eae396a4`) IS the batch pre-pass referenced by plan
  item 7 — enable it at model level via `parallel_evaluation` for the remaining point-gen win.
- P2-13 partially landed (`147b564bc`): prolongate/memset fusion + redundant CPR syncs; effects are
  within run-to-run noise at these sizes, as predicted (~0.1–0.2s class).

### P2-10 experiment findings (2026-07-18 session 2)
- The bs2 `linsolv_amgx` wrapper path executes mechanically (upload/setup/solve run; verified with a
  BLOCK_JACOBI probe).
- AMGX's **default MIN_MAX coloring hangs** on the 1.12M-row bs2 matrix — any multicolor stage-2
  config must set `"matrix_coloring_scheme": "PARALLEL_GREEDY"` (+ `max_uncolored_percentage`).
- With that, MULTICOLOR_DILU (and BLOCK_JACOBI) run but the outer GMRES stalls at max_iters with a
  growing residual — the relaxation-as-outer-solver apply is either numerically insufficient for
  SPE10 stage-2 or semantically wrong through the C-API path (garbage/unscaled X). Next steps for
  the study: validate M^-1 b on a small case against a reference, and/or wrap the smoother in a
  1-level AMG shell; until then the exact cuSPARSE block-ILU(0) stays.

### P2-10 discriminating tests (2026-07-18 session 3)
- bs2 AMGX as a full **PBICGSTAB** stage-2: outer GMRES converges in 1–7 iterations → **the block
  matrix upload is correct** (well-residual convergence issues remain from the variable-accuracy
  inner solve; not a viable configuration, but a valid probe).
- **Scalar-expanded** (`DARTS_CPR_STAGE2=amgx_bs1`, `convert_to_bs1`) MULTICOLOR_DILU: fails
  identically to the block variant → NOT a well-diagonal-block singularity.
- Net discriminator: SpMV-only solvers (BICGSTAB) work on this upload; diagonal/L-U-split
  relaxations (Jacobi, DILU) return destabilizing applies in both block and scalar form.
  Row-ordering suspicion REFUTED by code inspection: the Jacobian columns are sorted with the
  diagonal in order (engine_base.cpp:95-150). **Corrected prime suspect: singular RAW well-head
  diagonal entries** (e.g. zero dR/dz in a control row) — Jacobi/DILU invert raw diagonals and one
  Inf corrupts the Krylov basis (hence 'residuals' of 1e7 from a residual-minimizing method), while
  ILU0 survives because elimination modifies its pivots. Study route: an opendarts multicolor-DILU
  stage-2 kernel with well rows special-cased (identity/direct) — for structured grids a geometric
  8-coloring makes this tractable — rather than fighting AMGX per-row semantics.
- P2-13c config probes (3650d, reuse active): `min_coarse_rows=2048` exactly neutral (22.56s);
  `max_levels=5` catastrophic (52.1s — DENSE_LU coarse solve explodes). The default AMG hierarchy
  shape is near-optimal; coarse-level launch overhead is hidden under concurrent work. Item closed —
  no further gains from stage-1 config surgery.
