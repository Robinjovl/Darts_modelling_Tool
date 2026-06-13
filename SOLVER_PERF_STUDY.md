# MR280 Solver Performance / Migration Study

The gating study for workstream **F4** (dropping the proprietary non-ODLS CI jobs) and the
default-solver policy — see `MR280_MASTER_PLAN.md` §E. Measured **2026-06-12** on oahu
(48-core, 2× A100 80GB, CUDA 13.2).

**Builds / environments**
- MR280 (this branch, sha `da24fc39` + this working tree's fixes): repo
  `/oahu/data/avnovikov/open-darts-solvers`, env `solvers`, GPU build
  (`build_install_darts_gpu.sh --amgx --cudss`, OpenMP enabled).
- Legacy CPU BOS: repo `open-darts-chemistry` (darts-linear-solvers `56ce883`), env `chemistry`.
- Legacy GPU AMGX+CPR: repo `open-darts-development` (linear-solvers `56ce883`, 54 GPU engines),
  env `development`, A100.
- Harness: `/oahu/data/avnovikov/mr280_bench/` (`bench_cell.py` / `run_campaign.py` /
  `spe10_bench.py`; raw data `results.csv`). One fresh subprocess per cell, sequential
  execution, `OMP_NUM_THREADS=1` (CI parity) unless noted, solver identity verified per run
  from the engine's "Linear solver type" log line. NI/LI from `engine.stat`
  (`n_newton_total` / `n_linear_total`), wall = Python wall around `m.run()`.

---

## 1. Main comparison table

Cell = **NI / LI / wall**. CPU rows OMP=1.

| model | BOS legacy cpr_amg (chemistry) | MR280 FGMRES+CPR | MR280 default (model as-is) | MR280 MGR | SuperLU (direct) | Pardiso (direct) | GPU AMGX+CPR legacy (development) | MR280 GPU BiCGStab+ILU0 (in-tree) | MR280 GPU cuDSS (direct) |
|---|---|---|---|---|---|---|---|---|---|
| 2ph_do (100 c) | 90 / 267 / 0.12s | 88 / 88 / 0.13s | 88 / 88 / 0.13s | 88 / 289 / 0.12s | 89 / 89 / 0.11s | 88 / – / 0.39s | 89 / 204 / 3.04s | 89 / 184 / 2.59s | 88 / 88 / 1.54s |
| 3ph_bo (300 c) | 113 / 668 / 10.61s | 122 / 1712 / 9.09s | 122 / 1712 / 9.02s | 118 / 1584 / 11.38s | 129 / 129 / 10.34s | 129 / – / 10.46s | 132 / 583 / 21.78s | 125 / 1219 / 21.82s | 129 / 129 / 12.95s |
| 2ph_geothermal (500 c) | 215 / 275 / 0.39s | 215 / 215 / 0.37s | 215 / 215 / 0.36s | 215 / 492 / 0.27s | 215 / 215 / 0.25s | 215 / – / 0.66s | — | — | — |
| 2ph_comp (1000 c) | 1693 / 5036 / 7.87s | 2205 / 2205 / 11.35s | 2234 / 4188 / 9.04s (tuned MGR) | 2206 / 2206 / 9.15s | 2205 / 2205 / 9.46s | 2205 / – / 16.38s | 1693 / 3263 / 207.20s | 2205 / 2200 / 106.33s | 2205 / 2205 / 26.35s |
| 3ph_do (1000 c) | 70 / 196 / 0.53s | 70 / 70 / 0.61s | 70 / 70 / 0.61s | 70 / 70 / 0.54s | 70 / 70 / 0.54s | 70 / – / 0.94s | 70 / 139 / 9.32s | 70 / 70 / 3.92s | 70 / 70 / 1.52s |
| 2ph_do_thermal_mpfa | 194 / 1840 / 3.60s | 193 / 2853 / 6.28s | 193 / 2853 / 6.32s | 207 / 5175 / 7.30s | 193 / 193 / 9.65s | FAIL (pypardiso assert) | — | — | — |
| GeoRising (10.8k c) | 16 / 127 / 1.97s | 16 / 76 / 2.13s | 16 / 76 / 2.13s | **FAIL (>1800s timeout)** | 16 / 16 / 7.20s | 16 / – / 4.23s | 16 / 135 / 3.85s | 18 / 566 / 5.22s | 16 / 16 / 1.61s |
| Uniform_Brugge (~60k c) | 2364 / 5078 / 8.01s | **1194 / 1508 / 6.61s** | 1194 / 1508 / 6.62s | 2044 / 10731 / 6.93s | n/a (too large) | n/a | 2053 / 4091 / 208.84s | 2732 / 15978 / 210.90s | 1194 / 1194 / 117.28s |

Notes: Pardiso is Python-resident (LI not engine-counted). Direct solvers report LI = solver
calls (1/solve). The legacy-GPU walls on small models are dominated by AMGX/device init —
GPU is not meant for ≤60k-cell models.

Build-flavour caveat: CPU cells were measured on the CPU (MT) build; cells re-run on the GPU
build show ±1 NI on some models (e.g. 2ph_do CPR 88→89) because the GPU build compiles host
code with `-march=native` (FMA / FP reassociation). The CI CPU build does not use
`-march=native`, so CI references are unaffected.

## 2. SPE10 (1.122M cells, 60×220×85, 2ph dead-oil, 5 wells; run(50), OMP=8)

| solver | TS | NI | LI | wall | note |
|---|--:|--:|--:|--:|---|
| Historical proprietary BOS (`CPU_GMRES_CPR_AMG`, OMP=16, plan Appendix C) | 13 | 34 | 153 | ~45 s | reference point, different threading |
| Historical pre-BCSR-CPR MGR (plan Appendix C) | — | 41 | 1746+ | **terminated at T=13.6 after 26:39** | the stale 44× number |
| **MR280 MGR (tuned BCSR-CPR profile, this study)** | **13** | **33** | **209** | **166 s** | **iteration counts at BOS level; gap closed from 44× to per-iter/threading-level** |
| MR280 FGMRES+CPR | 13 | 32 | 2664 | 1276 s | correct (same Newton path as MGR) but 12.7× MGR's LI, 7.7× wall — confirms MGR's tuned profile as the SPE10-class choice |
| MR280 GPU (AMGX request → BiCGStab+ILU0 redirect) | 27 (12 wasted) | 104 (120 wasted) | 12921 | 423 s | completes, but ILU0-only prec takes heavy timestep cuts on SPE10's contrast — the `linsolv_bos_cpr_gpu` block-CSR port (AMG on pressure) is the needed follow-up for GPU heavy models |
| MR280 GPU cuDSS (direct, 2.24M scalar rows) | — | — | — | infeasible | factorization setup fails (memory: ≈40 GB free on the shared A100) — **but the failure path is clean**: 108 setup attempts each returned −1 → timestep cuts → graceful min-dt stop, no crash. Confirms direct solvers stay a ≤~200k-row option; SPE10-class stays iterative. |

## 2b. The GPU-CI 2-hour-timeout fix, validated

`fracture_network case_1` (Geothermal DFM) on the GPU platform: formerly the model's CPU-only
`SuperLUSolverSpec` was silently ignored on GPU and the iterative default hung → the whole GPU
test job died at the 2 h wall. With the platform-aware `set_solver()` (GPU → in-tree
`GPUCuSolverSpec` direct QR), the case now **completes on GPU in 106 s** (CPU: 12.8 s with
SuperLU). The GPU CI job regains ~1.5 h of budget.

## 3. Verdicts vs the pre-registered criteria (±15% wall vs legacy BOS)

| model | Δwall | ΔNI | verdict |
|---|--:|--:|---|
| 2ph_do | +4% | −2% | **PASS** |
| 3ph_bo | **−14%** | +8% | **PASS** (LI +156% but cheaper iterations) |
| 2ph_geothermal | −5% | 0% | **PASS** |
| 3ph_do | +14% | 0% | **PASS** |
| GeoRising | +8% | 0% | **PASS** (LI −40%) |
| Uniform_Brugge | **−17%** | **−49%** | **PASS — substantially better** |
| 2ph_comp | +44% | +30% | **ACCEPT?/investigate** — driver is +30% NI (different Newton path, i.e. solve-quality difference, not solver speed); wall via tuned-MGR default is +15% |
| 2ph_do_thermal_mpfa | +75% | 0% | **INVESTIGATE** — MPFA wide stencil; CPR per-iteration cost; (was an outright hang before the true-IMPES fix) |

**Aggregate: 6/8 PASS, 0 catastrophic.** The two open models have identified, scoped causes
(2ph_comp Newton-path; MPFA CPR tuning) — they are solver-tuning follow-ups, not architecture
problems.

## 4. Category conclusions

* **CPU iterative — keep FGMRES+CPR as the default.** It matches or beats the legacy BOS
  stack on 6 of 8 models (dramatically on Brugge) and never diverges.
* **MGR is not a viable general default** — diverges on GeoRising (>30 min vs 2.1 s), 2×
  NI on Brugge, worse on MPFA — but with the tuned BCSR-CPR profile it is the **SPE10-class
  heavy-model champion** (iteration parity with proprietary BOS at 1.1M cells). Keep it as
  the recommended per-model/tuned option, exactly as the MR description states.
* **Direct solvers** — SuperLU(CPU) and cuDSS(GPU) produce identical Newton paths
  (exactness cross-check passed on every common model). cuDSS extends the direct option to
  matrices CPU SuperLU can't touch (Brugge 180k scalar rows in 117 s) and is the robust
  fallback where iterative solvers diverge (GeoRising 1.6 s). At ≤1000 cells CPU SuperLU is
  ~3× faster than cuDSS (transfer + per-Newton refactorization overhead).
* **GPU iterative** — the in-tree open-source GPU story is currently BiCGStab+cuSPARSE-ILU(0):
  on Brugge it matches the legacy AMGX+CPR wall (210.9 vs 208.8 s) despite 4× LI. True
  AMGX-CPR in-tree is blocked on the `linsolv_bos_cpr_gpu` block-CSR port (see §5) — until
  then the legacy `development` build remains the GPU iterative reference.
* **Legacy-state columns** are captured for both requested baselines: BOS CPU
  (`cpu_gmres_cpr_amg` — note: `cpu_gmres_fs_cpr` is poromechanics-only and *errors* on flow
  engines; per-engine default decision recorded in the master plan) and GPU AMGX+CPR from
  `open-darts-development`.

## 5. Gate decision input (for F4 — dropping non-ODLS CI jobs)

Supports proceeding, with conditions:
1. Regenerate the `_odls` reference pkls from this build (the `_iter` refs embed proprietary
   Newton paths; the +30% NI on 2ph_comp etc. are mostly reference-provenance, not regressions).
2. Close or explicitly accept the two open models (2ph_comp, MPFA) — tracked in
   `MR280_MASTER_PLAN.md` workstream E.
3. The GPU CI axis (GSELINSOLVERSPATH) is untouched by F4; switching GPU CI to the in-tree
   stack should wait for the `linsolv_bos_cpr_gpu` block-CSR port.

Raw data: `/oahu/data/avnovikov/mr280_bench/results.csv` (+ `logs/`, campaign configs).
