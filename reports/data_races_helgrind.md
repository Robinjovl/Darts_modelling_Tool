# Helgrind (Valgrind) cross-check: `models/cpg_sloping_fault` (geothermal, `generate_5x3x4_wperiodic`, `OMP_NUM_THREADS=4`)

Second, independent tool run against the same model/case as `reports/data_races.md`
(ThreadSanitizer), specifically to separate genuine bugs from suspected
TSan/`libgomp`-annotation artifacts identified there.

## How this was produced

- Build: `build_valgrind/` — `CMAKE_BUILD_TYPE=Debug`, `ENABLE_VALGRIND=ON`
  (auto-applies `-march=x86-64-v2 -mno-avx512f`, needed so Valgrind's JIT
  doesn't SIGILL on AVX-512), `OPENDARTS_CONFIG=MT`. No sanitizer — plain
  Debug build, reusing the existing (non-instrumented) `thirdparty/install`
  HYPRE/SuperLU.
- Run:
  ```
  OMP_NUM_THREADS=4 valgrind --tool=helgrind --history-level=approx \
    --error-limit=no --num-callers=20 --log-file=helgrind_report.log \
    python3 -c "from main import run; run(physics_type='geothermal', \
    case='generate_5x3x4_wperiodic', out_dir='...', export_vtk=False)"
  ```
- Killed after ~10 minutes (Helgrind's overhead is very large, as expected).
  In that time it actually got *further* into the simulation than the earlier
  TSan run did in 5 minutes (multiple report steps, including a
  `FAILED TO CONVERGE` / re-cut timestep) and produced a 23 MB log:
  **2275 unique "Possible data race" contexts**, 44.8M total flagged accesses
  (Helgrind counts every conflicting access pair it hits at runtime, so this
  number scales with iteration count, not with the number of distinct bugs —
  don't read it as "44.8M bugs").

## Finding 1 — strong independent corroboration of the "unannotated libgomp barrier" hypothesis

`reports/data_races.md` flagged `engine_base::init_jacobian_structure`'s two
`#pragma omp parallel` first-touch regions (`engine_base.cpp:71` and `:119`)
as a likely ThreadSanitizer/`libgomp` false-positive: TSan reported the
master thread's post-barrier code racing against a worker thread's
pre-barrier `numa_set`/`memset`, even though the OpenMP standard guarantees
the barrier at the closing `}` of `#pragma omp parallel` must have already
joined every worker.

Helgrind — a completely different tool, using dynamic binary instrumentation
rather than compile-time sanitizer passes — flags **the exact same source
line** (`engine_base.cpp:71`) with the same shape:

```
Possible data race during write of size 4 ... by thread #7
   at ... libgomp.so.1 ... (worker, inside the region spawned at engine_base.cpp:71)
This conflicts with a previous access by thread #1, after
   (thread #1 created thread #7 via pthread_create, from GOMP_parallel at engine_base.cpp:71)
but before
   ... libgomp.so.1 ... engine_base.cpp:71 (again)
```

Two independently-implemented race detectors flagging the identical code
location with the identical "worker vs. spawning/joining thread" shape is
much stronger evidence that this specific report is a **tool/runtime
limitation** (neither TSan nor Helgrind fully understands GCC `libgomp`'s
internal barrier/thread-pool synchronization) rather than a real bug in
`init_jacobian_structure`. This doesn't retroactively prove *every* similarly-shaped
report in either tool's output is a false positive, but it's a solid existence
proof for the category.

## Finding 2 — new: HYPRE's own OpenMP-enabled internals are the single largest source of races

Not visible in the TSan pass (whose 5-minute window never got the linear
solver running against a large-enough problem to spend much time in the
preconditioner). Once Helgrind's slower run reached `solve_linear_equation()`
→ CPR/BoomerAMG setup, the *majority* of the 2275 contexts are inside HYPRE
itself, not open-DARTS code:

| Count | Location |
|---|---|
| 297 | `hypre_IJMatrixSetValuesParCSR` |
| 171 | unresolved `libgomp.so.1` frames |
| 142 | `hypre_BoomerAMGBuildCoarseOperatorKT` (`._omp_fn.5`) |
| 85 | `memmove` |
| 80 | `hypre_BoomerAMGBuildCoarseOperatorKT` (`._omp_fn.4`) |
| 65 | `hypre_BoomerAMGCreateSHost` (`._omp_fn.1`) |
| 60 | `hypre_BoomerAMGBuildStdInterp` |
| 49 | `hypre_IJMatrixAssembleParCSR` (`._omp_fn.0`) |
| 43 | `hypre_BoomerAMGBuildCoarseOperatorKT` |
| 42 | `hypre_BoomerAMGCoarsenPMISHost` |
| 41 | `hypre_BoomerAMGRelaxHybridGaussSeidel_core` (`._omp_fn.2`) |
| 40 | `hypre_CSRMatrixTransposeHost` (`._omp_fn.0`) |
| 37 | `hypre_CSRMatrixMatvecOutOfPlaceHost` (`._omp_fn.36`) |
| ... | (long tail of further `hypre_BoomerAMG*`/`hypre_CSRMatrix*` `._omp_fn.N` entries) |

This is directly relevant to `a5c8b1ac "enable openMP in hypre compilation"`
(recent commit on this branch's history, noted as a deliberate, cautious
experiment: *"we run CPU pipelines using nt=1 so this should not affect the
result ... on GPU pipelines we run with multiple threads ... let's try this
and check"*). This run is exactly that check, and the result is: with
`HYPRE_ENABLE_OPENMP=ON` and `nt=4`, Helgrind considers BoomerAMG setup
(coarsening, interpolation operator construction, IJ matrix assembly) and
CSR matvec to be extensively racy. I have **not** individually verified each
of these against the "unannotated barrier" pattern from Finding 1 — given the
volume, that would need to be done systematically (e.g. by writing a
suppression file for the confirmed-artifact shape and seeing what's left) —
but given HYPRE is a large, independently-maintained C library whose OpenMP
paths are used here in a way (many small `nt=4` solves back-to-back, one per
Newton iteration) that stresses exactly the region-boundary/thread-pool-reuse
pattern from Finding 1 far more than open-DARTS's own code does, it would not
be surprising if most of these are the same class of artifact rather than
distinct HYPRE bugs. This needs the same "does it reproduce under a
TSan-annotated OpenMP runtime" test as everything else before being trusted
either way.

**Follow-up (see `reports/data_races_clang_libomp.md`'s corrected Result 5
and `reports/gmres_race_root_cause.md`):** the hedge above was right. Tracing
representative examples across the largest categories found the same
barrier-boundary shape in 60-85% of cases, confirmed with a from-scratch
minimal reproduction that this shape is a proven ThreadSanitizer+OpenMP
limitation independent of tool or compiler. Read this finding as mostly
artifact, not a confirmed bug, with a residual, unquantified fraction not
individually ruled out either way — see Next steps.

## Finding 3 — the interpolator cache shows up again, but this specific hit is *also* stack-shaped

`reports/data_races.md`'s Category 1 (`multilinear_adaptive_cpu_interpolator`'s
`point_data`/`hypercube_data` caches, unguarded by any lock) recurs here too
(`materialize_missing_cache`, `multilinear_adaptive_cpu_interpolator.tpp:266-279`,
15 contexts). However, the specific example I traced in full here turned out
to be **`Address ... is on thread #1's stack`** — i.e. this particular hit is
comparing the current call's parallel-loop read against a completely
unrelated, already-finished, single-point evaluation from earlier in
`engine.init()` (`ms_well::initialize_control_epm` → `well_control_iface::initialize_well_block`),
reusing the same stack slot. That's the *same* barrier-boundary shape as
Findings 1/2, not the heap-based `std::unordered_map` mutation race that made
Category 1 the one TSan finding I was confident was a real, independent-of-any-tool-limitation
bug.

**This means Category 1 needs re-examination, not blanket confirmation.** Some
of its 250-ish TSan reports were on heap-allocated hash-table storage (a
stronger signal — see `reports/data_races.md`), but I have not gone through
enough of the Helgrind hits on this same code to say whether *those specific
heap-based* TSan reports are corroborated here too, or whether Helgrind
mainly caught the stack-shaped ones for this code path. Treat Category 1's
"high confidence there's no lock" as still true (that's a fact about the
source code, not a inference from either tool), but dial back "confidence
it's genuinely exercised concurrently" until this is checked more carefully.

## Finding 4 — one open-DARTS-owned race not prominent in the TSan sample

`opendarts::linear_solvers::(anonymous namespace)::block_csr_spmv_add`
(`linsolv_gmres.cpp:67`, `[clone ._omp_fn.0]`) — 15 contexts. Same file as the
`dot`/`axpy`/`scale` races already discussed with you in chat (all attributed
there to the same stack-reuse artifact). Given `block_csr_spmv_add` follows
the identical "small helper, its own `#pragma omp parallel for`, called
repeatedly from the same call site" shape as `dot()`, I'd bet on the same
explanation rather than a distinct bug, but haven't traced a full example for
this specific function yet.

## Summary vs. the TSan report

| | TSan | Helgrind |
|---|---|---|
| Coverage reached | ~20 nonlinear iterations, 5 min | further (multiple report steps, one re-cut timestep), ~10 min |
| Unique reports | 284 | 2275 |
| `engine_base::init_jacobian_structure` (barrier-boundary shape) | yes | yes — same line, same shape |
| Interpolator cache (`point_data`/`hypercube_data`) | yes, some heap-based | yes, but traced example was stack-based |
| HYPRE/BoomerAMG internals | not reached in the 5-min window | dominant category (>1000 of 2275) |
| GMRES vector kernels (`dot`/`axpy`/`scale`/`block_csr_spmv_add`) | yes (stack-based) | yes (`block_csr_spmv_add`) |

## Next steps (unchanged recommendation, now with two tools agreeing on it)

The single most useful next step is still: **re-run under a TSan-annotated
OpenMP runtime (Clang + LLVM `libomp`)**. Two tools independently reproducing
the same barrier-boundary shape at `engine_base.cpp:71` makes it more likely,
not less, that both are hitting the same real gap in how GCC's `libgomp`
exposes its synchronization to *any* dynamic analysis tool — which is exactly
the kind of thing a `libomp` build would settle, since `libomp`'s barriers
carry the annotations both tools need. Whatever still shows up under
Clang+`libomp` is real; whatever disappears was this artifact.

Full raw log kept at (session scratchpad, not in the repo):
`helgrind_report.log` (23 MB, partial — run was stopped after ~10 minutes) /
`helgrind_run_stdout.log`.
