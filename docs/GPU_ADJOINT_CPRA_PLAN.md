# Plan: adjoint gradients for the GPU super engine + CPRA preconditioning on GPU

Goal: bring the adjoint-gradient capability of `engine_super_cpu` (history matching /
optimization gradients via the discrete adjoint) to `engine_super_gpu`, and provide a
transpose-capable preconditioned Krylov stack (CPRA) on the GPU so the adjoint solves run at
GPU speed. Everything below is grounded in the current code (branch `xiaoming/add-mgr`);
citations are file:line.

## Status (2026-07-07): Phases 0 and 2 IMPLEMENTED

- **Phase 0 done + validated.** The CPU assembly body was extracted verbatim into the shared
  header `engines/src/engine_super_adjoint.hpp` (`super_engine_adjoint_assembly<Engine>`; all
  touched state lives on `engine_base`, both engines expose the identical operator-layout
  constants); `engine_super_cpu::adjoint_gradient_assembly` delegates to it. The
  opt-history-matching allocation blocks were extracted from `engine_base::init_base` into
  `engine_base::init_adjoint_base()` / `init_customized_operator_base()` and are now also
  called from `engine_base_gpu::init_base`. `engine_super_gpu::adjoint_gradient_assembly`
  re-anchors the device state to the historical X (uploads X + the host-re-evaluated
  op_vals/op_ders), re-runs the device Jacobian assembly, mirrors the values to the host and
  delegates to the shared helper. Fix on the way: `engine_base_gpu::evaluate_operators_d` now
  skips op sets with an empty block list (host-only customized operators would otherwise hit
  the CPU evaluator's unimplemented device entry).
  **Validation**: CPU superlu/mgr/cpra all reproduce angle 0.000614 after the refactor
  (unchanged); GPU superlu/mgr/cpra all run end-to-end and the GPU adjoint gradient is
  IDENTICAL to the CPU adjoint gradient (relative difference 3.9e-8, angle 0.00000 deg). The
  0.0242 deg adjoint-vs-numerical angle on GPU is entirely the finite-difference reference
  moving with the GPU forward solver (the FD gradients differ CPU-vs-GPU by 4.3e-4 while the
  adjoint gradients coincide), i.e. Phase 0 meets its gate.
- **Phase 2 implemented** (native CPRA-GPU stack): transposed device SpMV
  (`csr_matrix_base::{matrix_vector_product_t_d0, calc_lin_comb_t_d, refresh_transpose_spmv_d}`
  defaulted virtuals; `block_csr_matrix` serves them via `gpu_bsr_spmv`'s scalar-CSR view +
  generic `cusparseSpMV(TRANSPOSE)`); `linsolv_cusparse_ilu::solve_transposed` (transposed
  bsrsv2 pair on the shared factors, U^T then L^T, lazy analyses);
  `linsolv_bos_cpr_gpu::solve_transposed` (CPRA order, weights + sign mults on the
  prolongation side, P^T via a one-time host transpose map + device gather kernel, a SECOND
  pressure-preconditioner instance bound to P^T (`set_p_system_prec_t`) with lazy activation +
  per-setup refresh); `linsolv_gmres_gpu::solve_transposed` (forward algorithm with A^T SpMV
  and prec->solve_transposed; host-pointer staging for the host-side backward driver).
  Wiring: `engine_base::set_adjoint_solver_cpra_gpu()` virtual (implemented by
  `engine_super_gpu` under AMGX, bound to Python), `Adjoint_super_engine` gains
  `--adjoint-solver cpra-gpu` + `--platform gpu`.
  **Validation**: the cpra-gpu adjoint gradient equals the CPU reference adjoint gradient to
  3.8e-8 relative (angle 0.000001 deg) on the Adjoint model -- the native GPU transpose stack
  is numerically exact; the adjoint-vs-FD angle (0.0106 deg) is again the FD reference moving
  with the GPU forward.
- Found & fixed on the way: `linsolv_amgx::init` leaked its AMGX matrix/vector handles on
  repeated init (the adjoint backward driver re-inits its solver chain every gradient
  evaluation); the leaked objects outlived `AMGX_finalize` and AMGX's atexit MemManager
  aborted the process at exit. init() now destroys existing handles first, and the destructor
  tolerates never-initialised instances (null handles).

### Adversarial review round (2026-07-07) — repeated-init resource leaks fixed

A multi-agent adversarial review of the Phase 0+2 diff confirmed a *class* of resource leaks
that the single-evaluation validation could not surface: the adjoint backward driver calls
`linear_solver_ad->init()` once per gradient evaluation (engine_base.cpp — attach-time init +
one per `calc_adjoint_gradient_dirac_all`), so any solver in the CPRA-GPU chain that reallocates
in `init()` without freeing the prior allocation leaks on every optimization step and OOMs a
real (multi-evaluation) history-matching run. Fixed:
- **`linsolv_cusparse_ilu::init`** (HIGH): never freed `values_d_ilu`/`d_z`/`pBuffer`/descr_M,L,U/
  info_M,L,U before reallocating. Now factors the destructor into `free_device_resources()`
  and, at the top of init(), **frees the old resources then does a full fresh init**
  (destroy-then-recreate, the same pattern as `linsolv_amgx`). NB: an earlier attempt to *skip*
  re-allocation when the structure looked unchanged was **wrong and crashed** — the cached device
  row/col pointers and the bsrsv2 analyses go stale across the forward run; re-fetching them (a
  full fresh init) is required. The buffers are solver-internal so freeing them dangles nothing.
- **`linsolv_bos_cpr_gpu::init`** (HIGH): `P->init_device()` re-allocated the pressure-matrix
  device buffers + a cuSPARSE handle with no re-init guard. The pressure structure is FIXED
  across re-inits, so init now allocates the device mirror **once** (`if (P->gpu_mode != 1)`)
  and reuses it — this avoids both the leak and a **dangling reference** (an earlier
  `P->free_device()` fix freed buffers the pressure AMGX matrix still held, crashing the forward
  solve). Also extended the free-before-realloc block to reclaim `inverse_fail_counter`, the
  non-setup_gpu `p_vals`, and the host `P_B_h`/`P_X_h` buffers (LOW), and the destructor now
  `delete[]`s `P_B_h`/`P_X_h`. Distinction that matters: `P` is cpr-owned with fixed structure
  (skip-reuse is safe); the cusparse buffers alias `A`'s live device pointers (must re-fetch).
- **`engine_base::init_adjoint_base`** (MEDIUM): the four adjoint host matrices
  (`dg_dx_T`/`dg_dx_n`/`dg_dT_general`/`dT_du`) were `new`'d unconditionally, leaking on a
  repeated engine `init()` (now reachable on the GPU engine). Guarded the allocation
  (`if (!ptr) new`) while keeping `->init()` unconditional, matching the existing
  `dg_dx_n_temp`/`linear_solver_ad` idiom (mesh-change-safe).

Validation of the leak fixes: `models/Adjoint_super_engine/leak_probe.py` runs the forward once
then loops the adjoint gradient 15× while sampling per-process GPU memory — **flat at 1206 MB
(0 MB growth over 15 evaluations)** and the gradient is bit-stable (spread 4.2e-9), where before
the fix GPU memory grew every evaluation. `cpra-gpu` still returns a valid gradient (angle 0.009,
status 0, clean exit) and the leak fixes are all first-init-inert so the SPE10 forward stays
bit-identical.

Known minor, not changed (LOW, documented): the GPU backward pass assembles the device Jacobian
twice per timestep — the shared driver's forward `assemble_jacobian_array` runs first (against a
stale `X_d`, and its result is never mirrored to host on GPU), then
`engine_super_gpu::adjoint_gradient_assembly` re-anchors `X_d`/op arrays and re-assembles
correctly. Removing the redundant first assembly would require engine-specific dispatch in the
shared CPU/GPU driver; the wasted work is one cheap kernel launch, negligible next to the adjoint
solve, so it is left as-is to avoid risking the CPU path.

### Phase 1 (2026-07-07) — device adjoint assembly IMPLEMENTED

The per-cell/per-connection adjoint assembly was ported to CUDA and made Python-selectable:
- `engine_super_adjoint.hpp` was split into `super_engine_adjoint_assembly_host_loops` (the
  expensive A+B loops filling the block `dg_dx_n_temp` diagonal blocks + the scalar
  `dg_dT_general` values) and `super_engine_adjoint_finalize` (well-head handling + the
  `to_nb_1`/`build_transpose` scalar expansions). `super_engine_adjoint_assembly` runs both
  (unchanged host path for the CPU engine and the GPU host path).
- New `adjoint_gradient_assembly_kernel` (engine_super_gpu.tpp, thread per block row) reproduces
  `super_engine_adjoint_assembly_host_loops` term-by-term: the diagonal `dg/dx^n` (acc +
  kinetics), the convective `dg/dT` with sign-based upwinding, the diffusion `dg/dT` (mass vs
  energy, GRAD read as `c*NP+p` **exactly as the host adjoint**, NOT the forward kernel's
  `p*NE+c`), the rock-conduction `dg/dT`, the `count = N_VARS*(rows[i]-i)` /
  `c*N_element*+temp_num` write layout, and the `-=` accumulation order. No atomics (each thread
  owns a disjoint output region). The caller `cudaMemset`s both device buffers to 0 first (the
  kernel only writes diagonal `dg/dx^n` blocks and accumulates `dg/dT`).
- `engine_super_gpu::adjoint_gradient_assembly` branches on `engine.adjoint_assembly_on_gpu`
  (default true, bound to Python): device path = memset + kernel + copy the value arrays back to
  the host `csr_matrix` + shared `finalize`; host path = the reference loop. Buffers
  (`dg_dx_n_temp_values_d`, `dg_dT_general_values_d`, `conn_index_to_one_way_d`) are allocated /
  uploaded once in `init` under `opt_history_matching`.

**Validation** (Adjoint_super_engine, cpra-gpu, device vs host on the identical history):

| grid | cells | assembly device | assembly host | speedup | device-vs-host grad (rel) |
|---|---|---|---|---|---|
| 5x5x2 | 50 | 0.68 ms | 0.20 ms | 0.29x (overhead-bound) | 4.2e-14 |
| 40x40x10 | 16 000 | 2.9 ms | 80 ms | **28x** | 9.9e-10 |
| 80x80x20 | 128 000 | 20.6 ms | 732 ms | **35.6x** | 1.0e-9 |

The device assembly is a faithful reproduction of the host reference (gradient match 4e-14 at the
shipped 50-cell model, ~1e-9 at 128k cells -- the growth is FMA accumulation over more terms,
well within adjoint tolerance and comparable to the forward GPU-vs-CPU spread). It is only slower
than the host at trivial sizes (kernel-launch + copy-back overhead); by 16k cells it is ~28x
faster and the host assembly (which grows linearly, 0.73 s/gradient at 128k) is a real
optimization-loop cost the device path removes -- the whole backward pass drops from ~7.0 s to
~5.1 s per gradient at 128k cells (the remainder is the on-device adjoint solve + the host OBL
re-evaluation, the latter being the next target per Phase 3).

Phase 0's host assembly remains the selectable reference (`adjoint_assembly_on_gpu = False`).

**Adversarial review of the port (2026-07-07).** A multi-agent term-by-term review confirmed the
non-thermal port is exact and surfaced:
- **Device buffer leak (fixed):** `~engine_super_gpu` freed only `mesh_grav_coef_d`; the three new
  adjoint buffers (and the sibling mesh `_d` arrays) leaked on engine destruction. The destructor
  now frees all of them (the adjoint buffers are nullptr-initialised, so freeing is a safe no-op
  when `opt_history_matching` is off).
- **THERMAL adjoint is incomplete (pre-existing, NOT introduced by the port; documented).** For
  `THERMAL=true` the energy-equation rows of both `dg/dx^n` and `dg/dT` omit terms the forward
  Jacobian carries: the rock- and potential-energy accumulation derivatives on the `dg/dx^n`
  diagonal, and the potential-energy convective flux + the fickian-enthalpy advection on `dg/dT`.
  These omissions live in the **host reference** (`super_engine_adjoint_assembly_host_loops`) and
  the device kernel reproduces them *identically* -- i.e. the port is faithful (device == host);
  the gap is a property of the host adjoint, not the CUDA port. Separately, the reused forward
  `assemble_jacobian_array_kernel` has a pre-existing store-vs-`atomicAdd` race on the energy row
  (the `c==NC` thread's direct store can clobber the `c<NC` fickian-enthalpy `atomicAdd`s), which
  corrupts the thermal forward Jacobian (and hence the thermal adjoint that consumes it). Fixing
  the thermal adjoint requires completing BOTH the host reference and this kernel together (to keep
  device == host) plus resolving the forward-kernel race -- tracked as follow-up, out of scope for
  "port the existing host assembly to CUDA". The non-thermal path (the validated + benchmarked
  configuration) is unaffected.

## 1. How the CPU adjoint works today (what must be reproduced)

Data flow per optimization gradient evaluation:

```
FORWARD (per converged step, opt_history_matching=true):
  engine_base::post_newtonloop                       engine_base.cpp:3234-3264
    X_t / dt_t / t_t / well_control_arr push_back    (host history)

BACKWARD (host driver, non-virtual):
  Python: OptModuleSettings.grad_adjoint_method_all  darts/models/opt/opt_module_settings.py:1859-1981
    -> engine.calc_adjoint_gradient_dirac_all()      engine_base.cpp:449-835
  for ts = last .. first:
    restore X, dt, wells from history
    HOST OBL re-eval: evaluate_with_derivatives -> op_vals_arr / op_ders_arr   (:652-657, :750-755)
    assemble_jacobian_array(dt, X, Jacobian, RHS)    (virtual -> engine-specific)
    prepare_dj_dx -> Temp_dj_dx, Temp_dj_du          (host, well-control AD operators)
    adjoint_gradient_assembly(dt, X, ...)            (virtual; CPU: engine_super_cpu.tpp:839-1240)
       fills dg_dx_n_temp (block, diagonal-only)     accumulation/kinetics derivs vs X^n
             dg_dT_general ((nb*nv) x n_interfaces)  residual derivs vs per-connection trans
             dg_dx_T (scalar transpose of Jacobian)  ONLY when the adjoint solver does not
                                                     transpose itself (guard :1175)
    RHS_adj = Temp_dj_dx - dg_dx_n^T lambda_n        (:776-779)
    lambda = A^{-T} RHS_adj                          solver_ad.setup + solve_transposed/solve (:781-800)
    gradient += dg_dT_general^T lambda + Temp_dj_du  (:808-818, well-head trans stripped)
  gradient_u = dT_du^T gradient -> engine.derivatives (:824-827)
```

Two adjoint solver modes (`set_adjoint_linear_solver(solver, use_jacobian_transpose)`,
engine_base.h:227-249):
- `use_jacobian_transpose=true` (**mgr**, **cpra** modes in `Adjoint_super_engine`): the raw
  block `Jacobian` is passed and the solver performs the transpose internally through
  `solve_transposed()` (CPR: CPRA per Han et al. 2013, lazy chain; MGR: explicit transpose +
  re-setup, `requires_setup_for_transposed_solve()==false`). `dg_dx_T` is **never built**.
- `use_jacobian_transpose=false` (**superlu** mode): the scalar transpose `dg_dx_T` is
  materialised on the host (engine_super_cpu.tpp:1174-1206) and solved directly.

Key structural facts for the port:
- `calc_adjoint_gradient_dirac_all` and `prepare_dj_dx` are **non-virtual host code** in
  engine_base.cpp; only `assemble_jacobian_array` and `adjoint_gradient_assembly` are virtual.
- The backward loop re-evaluates OBL operators **on the host** each step — this works for any
  engine type as long as host `op_vals_arr`/`op_ders_arr` are filled (the driver fills them
  itself, engine_base.cpp:652/750).
- All observation plumbing, history storage and pybind bindings are on `engine_base` and are
  already inherited by the GPU engine (py_engine_super_gpu.cpp:27) — **no new Python API**.

## 2. GPU engine: what exists, what is missing

| component | state | citation |
|---|---|---|
| `adjoint_gradient_assembly` (GPU) | **stub returning 0** | engine_super_gpu.tpp:809-813 |
| forward assembly kernel (template for an adjoint kernel) | exists | engine_super_gpu.tpp:358 (kernel), :740-753 (launch) |
| `op_vals_arr_d` / `op_ders_arr_d` / `op_vals_arr_n_d` on device | exists (static + adaptive itors implement `evaluate_with_derivatives_d`) | engine_base_gpu.cpp:68-96; interpolation/*gpu*.tpp |
| mesh/PV/RV/tran/grav device mirrors | exist | engine_super_gpu.tpp:718-734 |
| device Jacobian (values/structure accessors) | exists | engine_base_gpu.h:149-180 |
| `dg_dx_T` / `dg_dx_n` / `dg_dT_general` on GPU path | **not allocated** (CPU-only, engine_base.h:1191-1215) | — |
| host Jacobian values after GPU assembly | **stale** (values live on device) | engine_base_gpu.cpp:184-187 |
| transpose SpMV on device | **missing** (`cusparseDbsrmv` supports only NON_TRANSPOSE; wrapper hardcodes it) | gpu_bsr_spmv.cpp:208 |
| `solve_transposed` on any GPU solver | **missing** (all inherit the −1 default) | linear_solver.hpp:165 |
| transposed CPR reduce/prolongate kernels | **missing** (forward-only) | linsolv_bos_cpr_gpu.cu:533,570 |
| cuSPARSE block-ILU transposed triangular solves | **feasible**: `bsrsv2` supports `CUSPARSE_OPERATION_TRANSPOSE`; currently hardcoded NON_TRANSPOSE | linsolv_cusparse_ilu.hpp:172-173 |
| AMGX transpose solve | **not offered by the AMGX C API** (only `AMGX_solver_solve*`) — an explicit transposed pressure matrix must be fed as an ordinary solve | thirdparty/AMGX/include/amgx_c.h:386-391 |
| scalar-CSR device view of the block Jacobian | exists (`build_scalar_csr_device`) — enables generic `cusparseSpMV` which DOES support transpose on CSR | block_csr_matrix.hpp:164-171 |
| well rows | host-assembled then scattered to device (pattern reusable for adjoint well rows) | engine_super_gpu.tpp:783-798 |

## 3. Design decisions

**D1 — keep the backward driver on the host.** `calc_adjoint_gradient_dirac_all` stays as-is:
it is non-virtual, deeply entangled with host observation data, well AD operators and history
vectors, and its per-step vector work is O(n) — not the bottleneck. What moves to the device is
the two expensive pieces: the adjoint **assembly** and the adjoint **solve**.

**D2 — target the `use_jacobian_transpose=true` mode first (mgr/cpra path).** It avoids
materialising `dg_dx_T` entirely: the solver consumes the (device-resident) forward Jacobian
and transposes internally. The legacy superlu/`dg_dx_T` mode gets a host-sync fallback (D5).

**D3 — transpose-on-GPU strategy per stage** (cuSPARSE BSR has no transposed SpMV and AMGX has
no transposed solve, so each stage needs its own approach):
- *Krylov transpose SpMV* (`A^T v` in GMRES-transposed): use the existing scalar-CSR device
  view (`build_scalar_csr_device`) with the **generic** `cusparseSpMV` and
  `CUSPARSE_OPERATION_TRANSPOSE`. No new matrix materialisation; structure cached, values
  re-gathered per setup (already implemented for AMGX bs-1 consumption).
- *ILU(0)^T stage*: reuse the existing `bsrilu02` factors; apply the transposed preconditioner
  as `(LU)^{-T} = U^{-T} L^{-T}` via `bsrsv2` with `CUSPARSE_OPERATION_TRANSPOSE` and swapped
  L/U order (new analysis infos for the transposed solves; factors unchanged).
- *Pressure AMG^T stage*: materialise the explicit transpose of the **scalar pressure matrix**
  A_p (size nb×nb, one entry per block — small next to the full system) with
  `cusparseCsr2cscEx2` on device, then run AMGX on A_p^T as an **ordinary** solve. Setup once
  per adjoint step; same lazy-activation pattern as the CPU CPRA chain (built on first
  transposed solve, refreshed on later setups).
- *CPR reduce/prolongate transposed kernels*: mirror the CPU CPRA operator order
  (linsolv_cpr.cpp solve_transposed: ILU^T → r − A^T x → pressure restrict (plain C^T) →
  A_p^T solve → prolong+add), implemented as two small kernels next to the existing forward
  `cpr_solve_reduce_kernel`/`cpr_solve_prolongate_kernel`. Note the forward True-IMPES weights
  enter on the *restriction* side in forward CPR and on the *prolongation* side in CPRA.

**D4 — adjoint assembly as a device kernel with host mirrors for the reductions.** New
`adjoint_gradient_assembly_kernel` in engine_super_gpu.tpp mirroring the forward kernel's
cell/connection loop structure (same inputs: `X_d, op_vals_arr_d, op_ders_arr_d, op_vals_arr_n_d,
PV_d, RV_d, mesh_tran_d, mesh_tranD_d, mesh_grav_coef_d, mesh_kin_factor_d`), writing:
- `dg_dx_n_diag_d` — the diagonal blocks of ∂g/∂xⁿ (the CPU version zeroes all off-diagonals,
  engine_super_cpu.tpp:1123 — so a dense (nb × nv²) diagonal array suffices; no CSR needed);
- `dg_dT_values_d` — the (n_conns × nv) values of ∂g/∂T (structure built once on host by the
  existing `init_adjoint_structure`, engine_base.cpp:337-431).
After the kernel: `cudaMemcpy` both value arrays to their host `csr_matrix` counterparts
(`dg_dx_n`, `dg_dT_general`) so the host driver's reductions (`dg_dx_n^T λ`, `dg_dT_general^T λ`,
well-head stripping) run unchanged. Copy volume per backward step ≈ nb·nv² + n_conns·nv doubles
(≈ 90 MB for full SPE10 at nv=2) — small next to the adjoint solve. Well rows: reuse the
host-assemble + device-scatter pattern (or zero the well rows on device exactly as the CPU loop
does, engine_super_cpu.tpp:1162-1172, and harvest `well_head_tran_idx_collection` on the host
from the mesh — it is structure-only and time-invariant).

**D5 — superlu / `dg_dx_T` fallback mode**: sync the device Jacobian values to the host
(`sync_to_host` on the values dual_array) and run the existing host `to_nb_1` +
`build_transpose` path. Correct but slow; kept as the verification reference and for
robustness, not the default.

**D6 — adjoint solver injection on GPU** reuses the machinery landed in the reconfiguration
work: `engine.set_adjoint_linear_solver(solver, use_jacobian_transpose=True)` already binds and
initialises against the live Jacobian (engine_base.h:227-249). The new GPU solvers plug in
through the same call; `Adjoint_super_engine`'s `cpra` mode gains a GPU variant that builds
`linsolv_gmres_gpu + linsolv_bos_cpr_gpu` with the transposed entry points.

## 4. Phased implementation

### Phase 0 — correctness first: GPU forward + host adjoint (small, unblocks users)
1. Allocate the adjoint matrices in the GPU engine's `init_base` under `opt_history_matching`
   (same code as engine_base.h:1191-1215; they are host `csr_matrix<1>` objects — reusable
   verbatim).
2. Implement `engine_super_gpu::adjoint_gradient_assembly` as a **host** implementation: sync
   `Jacobian` values device→host, and run the same loops as the CPU version. Do this by
   extracting the body of `engine_super_cpu::adjoint_gradient_assembly` into a shared host
   helper (template on N_VARS, parameterised by the op arrays) called by both engines — no
   copy-paste divergence.
3. Adjoint solve: CPU CPRA/MGR (host solvers on the host-synced Jacobian) — both already
   support `solve_transposed`.
4. **Validation gate**: `Adjoint_super_engine` runs on `platform='gpu'` with gradient angle
   equal to the CPU run (reference 0.000614°) for superlu, mgr, and cpra adjoint modes.

Effort: ~2-4 days. Result: GPU-accelerated forward simulations get correct gradients; the
backward pass runs at CPU speed.

### Phase 1 — device adjoint assembly
1. `adjoint_gradient_assembly_kernel` per D4 + device buffers `dg_dx_n_diag_d`,
   `dg_dT_values_d` (allocated once under `opt_history_matching`).
2. Host copy-back + population of `dg_dx_n` / `dg_dT_general` values (structure unchanged).
3. Reuse `evaluate_operators_d` on the restored `X_t[idx]` (copy state host→device, evaluate on
   device) so the backward step no longer needs the host OBL re-eval for the assembly — keep
   the host re-eval only for `prepare_dj_dx` (well-control operators, host AD etors).
4. Validation: bitwise comparison of `dg_dx_n` / `dg_dT_general` values vs the Phase-0 host
   assembly on 2ph_do and the Adjoint model; gradient angle unchanged.

Effort: ~1 week (kernel is a structural mirror of the forward one).

### Phase 2 — transpose solve stack on GPU (CPRA-GPU)
1. `gpu_bsr_spmv::spmv_transposed(x, y)` via the scalar-CSR device view + generic
   `cusparseSpMV(CUSPARSE_OPERATION_TRANSPOSE)`; wire
   `block_csr_matrix::matrix_vector_product_t_d` alongside the forward `_d` entry points.
2. `linsolv_cusparse_ilu::solve_transposed` — transposed `bsrsv2` pair (new analysis infos,
   swapped L/U order, same factors).
3. `linsolv_bos_cpr_gpu::solve_transposed` — CPRA operator order per D3 with:
   - transposed prolong/reduce kernels (weights on the prolongation side);
   - device transpose of the scalar pressure matrix (`cusparseCsr2cscEx2`, buffer cached);
   - a **second AMGX solver instance** bound to A_p^T (AMGX has no transpose API); lazy
     creation on the first transposed solve, `AMGX_matrix_replace_coefficients` on refresh —
     the same lazy-adjoint-chain pattern as the CPU CPR.
4. `linsolv_gmres_gpu::solve_transposed` — the forward algorithm with `A^T` SpMV and
   `prec->solve_transposed`, mirroring the CPU `linsolv_gmres` transpose flag; plus
   `requires_setup_for_transposed_solve()` handling.
5. Engine wiring: `Adjoint_super_engine` `cpra` mode on GPU builds this stack;
   `set_adjoint_linear_solver(..., use_jacobian_transpose=True)`.
6. Validation: adjoint gradient angle vs CPU CPRA on the Adjoint model; per-solve transposed
   residual checks (‖A^T x − b‖) in a unit test against a small dumped matrix.

Effort: ~1.5-2 weeks. This is also the enabling work for any future transpose need on GPU
(e.g. least-squares subproblems), not adjoint-specific.

### Phase 3 — performance & cleanup
- Move the two host reductions (`dg_dx_n^T λ`, `dg_dT_general^T λ`) to device SpMV if profiling
  shows them (unlikely: O(nnz) once per backward step).
- Batch the backward loop's host↔device state copies (X_t restore) with pinned memory.
- Optional: cuDSS as the exact adjoint reference on GPU — check `CUDSS_CONFIG` for a
  transpose-solve option; if absent, factor A^T explicitly via the csr2csc output (cuDSS is
  already wired for the scalar view).
- Retire the D5 host fallback from the default path once Phase 2 is validated.

## 5. Risks & mitigations

| risk | mitigation |
|---|---|
| AMGX on A_p^T converges differently than on A_p (non-symmetric pressure operator) | CPU CPRA has the same asymmetry and works (validated 0.000614°); keep MGR-transpose and the D5 superlu fallback as alternates |
| cusparse generic SpMV transpose performance (CSR^T = CSC access pattern) | acceptable per-iteration cost at adjoint frequencies; if hot, materialise the explicit A^T once per setup via csr2csc and use forward SpMV |
| divergence between CPU and GPU adjoint assembly formulas | Phase 0 shared host helper + Phase 1 bitwise value comparison |
| well-derivative rows subtly different on GPU | keep well rows host-computed (they are already host-assembled in the forward GPU path) |
| history growth (X_t) for long runs | unchanged from CPU behaviour; document `clear_previous_adjoint_assembly` |

## 6. Acceptance criteria

1. `Adjoint_super_engine` with `platform='gpu'`: gradient angle equals the CPU result
   (0.000614° on the reference case) for superlu (Phase 0), mgr (Phase 0), and cpra-GPU
   (Phase 2) adjoint modes.
2. No forward-path regression: SPE10 forward GPU benchmark unchanged (17 s class).
3. Backward wall-clock: Phase 1+2 target ≥ 5× speedup of the adjoint pass vs the Phase-0
   host fallback on a 1M-cell case (assembly + solve both on device).
4. Unit tests: transposed SpMV, transposed ILU apply, CPRA-GPU single-apply vs CPU CPRA on an
   identical small system.
