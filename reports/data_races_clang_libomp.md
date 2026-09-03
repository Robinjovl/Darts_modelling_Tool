# Clang + LLVM `libomp` TSan cross-check: `models/cpg_sloping_fault` (geothermal, `generate_5x3x4_wperiodic`, `OMP_NUM_THREADS=4`)

Third tool/runtime combination against the same model/case as `reports/data_races.md`
(GCC + `libgomp`, ThreadSanitizer) and `reports/data_races_helgrind.md`
(GCC + `libgomp`, Helgrind). This was specifically built to answer the
question those two reports left open: is GCC's `libgomp` (which lacks
ThreadSanitizer annotations for its own barriers) responsible for the
"barrier boundary" race reports, or are they real? Barrier-boundary reports
turned out to be artifacts across the board (confirmed and explained in
`reports/gmres_race_root_cause.md` — neither GCC's `libgomp` nor a standard
LLVM `libomp` build is TSan-annotated), while the non-barrier categories held
up, and one brand-new category emerged that neither earlier report could
reach.

## How this was produced

No system Clang/`libomp` was available (no root for `apt`), so this used a
self-contained conda-forge toolchain instead:

```
conda create -n darts-clang-tsan -c conda-forge python=3.11 clangxx=23.1.0 llvm-openmp=23.1.0 cmake make
```

**Everything** — HYPRE, SuperLU, and the main project — was rebuilt with this
toolchain, into fresh trees (`thirdparty/install_clang`, `build_tsan_clang/`),
so the whole stack (including HYPRE's own OpenMP-enabled internals) runs on
LLVM's `libomp` instead of GCC's `libgomp`, all with `-fsanitize=thread`.
Reusing the existing GCC-built `thirdparty/install` would have meant HYPRE
staying linked against `libgomp` while the rest of the process used `libomp`
— two different OpenMP runtimes loaded into one process, which is unsupported
and would confound the comparison (or just crash/deadlock).

Getting this to build surfaced **three real, previously-latent portability
bugs** — nothing to do with threading, just code that only ever saw GCC:

1. `CMakeLists.txt`: the Clang branch unconditionally passed
   `-Wl,-no_compact_unwind` to the linker. That flag is Apple `ld64`-only
   (added for AppleClang, per the inline comment); GNU `ld`/`lld` — what
   Clang uses on Linux — reject it outright ("unable to disambiguate").
   Fixed by gating it to `if(APPLE)`.
2. `discretizer/src/approximation.h:258-259`: `constexpr index_t nb1 = ap1.n_block;`
   accessed a static constexpr member through a dependent-type function
   parameter; Clang rejected the initializer as non-constant. Fixed by using
   the fully-qualified form (`LinearApproximation<VarNames1...>::n_block`),
   which is unambiguous under any compiler.
3. `discretizer/src/mech_discretizer.cpp:1039-1040`: explicit template
   instantiation (`template class MechDiscretizer<POROELASTIC>;`) appeared
   after only a `using namespace dis;`, not inside `namespace dis`. Per the
   standard, explicit instantiation of a namespace member must occur in that
   namespace (or an enclosing one) — a `using` directive doesn't count. GCC
   accepts this as an extension; Clang correctly rejects it. Fixed by
   wrapping the instantiations in `namespace dis { ... }`.
4. A fourth, more interesting one surfaced only once (3) was fixed and the
   code actually *ran* for the first time under Clang:
   `mech_discretizer.cpp:18` initialized a static data member via
   `N_UNKNOWNS.at(MODE)`, where `N_UNKNOWNS` is a *different* static
   (header-defined) `unordered_map`. Nothing in the standard guarantees that
   map is constructed before this initializer runs — a textbook
   static-initialization-order-fiasco. GCC's particular link order happened
   to get it right (silently, for years); Clang's didn't, and the process
   aborted with `terminate: unordered_map::at: out_of_range` on
   `import darts.discretizer`. Fixed by computing the value directly
   (`(MODE == THERMOPOROELASTIC) ? (ND + 2) : (ND + 1)`) instead of going
   through a runtime-constructed map during static init.

All four fixes are committed as regular source changes (uncommitted in git at
time of writing — see `git diff`), independent of the TSan investigation, and
worth keeping regardless of what happens with the race-hunting.

Run (once the build worked):
```
OMP_NUM_THREADS=4 LD_PRELOAD=$(clang++ -print-file-name=libclang_rt.tsan-x86_64.so) \
TSAN_OPTIONS="halt_on_error=0:history_size=7:log_path=tsan_report_clang" \
python3 -c "from main import run; run(physics_type='geothermal', \
case='generate_5x3x4_wperiodic', out_dir='...', export_vtk=False)"
```
Ran for ~18.5 minutes of CPU time before being stopped. Notably, this build
**ran much further** than either earlier attempt — it converged past
`T=730` (days) with clean Newton behavior (`NI=2, LI=5` for most steps),
versus the GCC builds which were still churning through repeated well
switching in the first ~20 iterations when their time windows ran out. Not a
claim about correctness of one build over the other — just explains why the
raw counts below aren't apples-to-apples with the shorter earlier runs.

**1108 unique "WARNING: ThreadSanitizer: data race" reports.**

## Result 1 — the `engine_super_cpu.tpp:153` self-race is gone

`reports/data_races.md` flagged a self-race at the very first line of
`assemble_jacobian_array`'s parallel region (`int id = omp_get_thread_num();`),
between two *unrelated* calls reusing the same stack slot
(`Location is stack of main thread`). Under Clang + `libomp`: **zero**
occurrences. This is a clean confirmation that this specific report was a
GCC-`libgomp`-annotation artifact, not a real bug.

## Result 2 — `engine_base::init_jacobian_structure`'s barrier-boundary race is sharply reduced, not eliminated

`engine_base.cpp:71` (the first-touch `#pragma omp parallel` region) went
from being a recurring category (in both the TSan and Helgrind GCC runs) to
**3 occurrences** here — `init_jacobian_structure` only runs once per
`engine.init()` call, so this isn't diluted by the longer run; it's a real
drop. Mostly consistent with the artifact theory, but not a clean zero, so I
wouldn't call this fully closed the way Result 1 is.

## Result 3 — the GMRES `dot`/`axpy`/`scale` races persist under `libomp` too

`reports/data_races.md` and the in-chat explanation both leaned on "GCC
`libgomp` doesn't annotate its barriers" to explain the `dot()`
reduction-buffer race (`linsolv_gmres.cpp:180` vs `:187`, reused stack
`partial[]` across sequential calls). Under Clang + `libomp` — the allegedly
TSan-aware runtime — **the exact same race reproduces** (62 occurrences in
the categorized sample, same two lines, same shape).

At the time this report was first written that looked like a problem for the
"GCC-specific artifact" theory. It wasn't: **`reports/gmres_race_root_cause.md`**
tracked this down with a minimal, standalone reproduction (no GMRES, no
engine code) and found the *exact same* false positive on the single most
canonical "manual OpenMP reduction into a per-thread stack slot" idiom,
under both GCC+`libgomp` and Clang+`libomp`, with a single parallel region
and no repetition. Neither prebuilt OpenMP runtime is instrumented with the
TSan annotation hooks needed to recognize its own team-join barrier as a
synchronization point — a documented, runtime-agnostic ThreadSanitizer+OpenMP
limitation, not an open-DARTS bug. Closed; see that report for the full
mechanism, and see the **correction to Result 5 below** — it turns out this
same mechanism explains most of the HYPRE findings too, which I originally
(wrongly) called a confirmed bug independent of any barrier question.

## Result 4 — the interpolator cache race is now confirmed across three independent tool/runtime combinations

`multilinear_adaptive_cpu_interpolator`/`point_data_store` races: **376
blocks** in this run (heavily present — this run went much further, so more
opportunities to hit it, but it was never close to zero). Checked a
representative example: `Location is heap block of size 760` — genuine
shared heap storage (a hash-table node), not a stack-reuse artifact. This is
now the third independent confirmation (GCC+TSan, Helgrind, Clang+TSan+libomp)
of the same finding: **the adaptive interpolator's `point_data`/`hypercube_data`
caches have no lock, and are exercised concurrently in a way that produces
real races.** I'd now call this a confirmed defect, not a suspected one —
see `reports/data_races.md` Category 1 for the code-level detail and the
fix recommendation (add real synchronization to those two maps).

## Result 5 (CORRECTED) — HYPRE's CPR/BoomerAMG "races" are mostly the same barrier artifact as the GMRES race, not a confirmed bug

**This section originally claimed a confirmed, compiler-independent HYPRE
thread-safety bug. That was premature — I hadn't yet checked whether the
HYPRE reports were barrier-boundary shaped the way the GMRES race turned out
to be (see `reports/gmres_race_root_cause.md`). Having gone back and traced
several representative examples in detail, most of them are.**

The dominant category in this run: `linsolv_cpr::setup_unguarded` (367),
`linsolv_cpr::solve_unguarded` (107), `hypre_ij::build`/`refresh` (43 + 9),
`set_hypre_vector` (20) — all inside the CPR preconditioner's HYPRE
setup/solve path. Traced examples:

- `linsolv_cpr::build_pressure_subsystem` reads a heap buffer; the "previous
  write" is a worker thread inside `assemble_jacobian_array`'s own,
  long-since-closed first-touch parallel region (`globals.h:184`) — a
  completely unrelated function, from an earlier point in the run, reusing
  the same heap address after its region's barrier already joined.
- HYPRE's own `hypre_PrefixSumInt`: a worker writes inside its `omp_outlined`
  parallel region (`prefix_sum.c:44`); the "conflicting" access is the main
  thread reading the same value immediately after, in serial code at
  `prefix_sum.c:54` — same function, same shape as `dot()`.
- Two *different* HYPRE routines called sequentially in the AMG smoothing
  cycle (`hypre_BoomerAMGRelaxHybridGaussSeidel_core` then
  `hypre_CSRMatrixMatvecOutOfPlaceHost`), each with its own closed parallel
  region, reusing the same scratch/residual buffer between calls.
- `hypre_prefix_sum_pair`: a worker's first write to a buffer racing against
  the buffer's own `malloc` on the main thread — same annotation gap, this
  time on the allocate-then-fork edge rather than fork-then-join.

A script over the largest categories found this "one access inside an
OpenMP-outlined region (its own or a different, earlier call's), the other
access outside it or inside a different, sequential call's region" shape in
**60-85% of cases** (`BoomerAMGBuildStdInterp`: 60/70, zero exceptions;
`BoomerAMGBuildCoarseOperatorKT`: 97/124; `BoomerAMGCoarsenPMISHost`: 71/103).
I also went back and checked a Helgrind example
(`hypre_IJMatrixSetValuesParCSR`, referenced in
`reports/data_races_helgrind.md` Finding 2) and it showed the identical
shape: the conflicting access was a worker thread inside
`engine_base::init_jacobian_structure`'s own already-closed parallel region
— the same finding I'd *already* flagged as a likely artifact there,
just not connected to the HYPRE numbers at the time.

**This doesn't fully close HYPRE the way `reports/gmres_race_root_cause.md`
closed the GMRES race.** 13-22 examples per category (up to ~37% for
`CSRMatrixMatvecOutOfPlaceHost`) show *both* sides inside OpenMP-outlined
frames rather than one side being cleanly serial. I checked one of these
(the Gauss-Seidel/matvec pair above) and it was still explainable as two
sequential calls sharing a buffer, not genuine simultaneous access — but I
have not exhaustively verified every one of them the way the from-scratch,
zero-HYPRE minimal reproduction ruled out any doubt for `dot()`. HYPRE's
AMG setup and solve is a long chain of small, separately-parallelized
internal routines called one after another, reusing scratch vectors and
matrices across V-cycle levels and Newton iterations — exactly the shape
that triggers this ThreadSanitizer+OpenMP limitation, and it triggers it a
lot simply because there are so many such routines chained together. That
architectural fact, on its own, is enough to produce the bulk of what was
seen here without HYPRE having any actual bug.

**Revised verdict: not a confirmed bug. Most likely the same
barrier-annotation artifact, at a scale mostly explained by how many
small internally-parallel routines HYPRE chains together — with a residual,
unquantified fraction (the "both sides in OpenMP regions" cases) not fully
ruled out either way.** This also means the `a5c8b1ac "enable openMP in
hypre compilation"` commit's "let's try this and check" experiment did not,
on this evidence, turn up a demonstrated problem for `nt>1` correctness —
though it also doesn't clear it; see Recommendation below for how to actually
settle this.

## Result 6 — `well_controls.cpp` / `assemble_jacobian_array` op-array races persist

`well_control_iface::add_to_jacobian` (33) and
`engine_super_cpu.tpp::assemble_jacobian_array` (32, excluding the
interpolator-cache-tagged ones already counted in Result 4) still show up.
Consistent with `reports/data_races.md` Category 2 — these read a neighbor
block's `op_vals_arr`/`op_ders_arr` while another thread's on-demand
interpolation call is still writing it. Given Result 4's now-confirmed
verdict on the interpolator cache, and that these entries are the same
underlying data path, I'd fold this into the same "real, needs fixing"
bucket rather than treating it separately.

## Summary across all three tool/runtime combinations

| Category | GCC+TSan | Helgrind (GCC) | Clang+TSan+libomp | Verdict |
|---|---|---|---|---|
| `engine_super_cpu.tpp:153` self-race | yes | — | **gone (0)** | barrier artifact (confirmed, see `gmres_race_root_cause.md`) |
| `engine_base.cpp:71` init first-touch | yes | yes (same line) | sharply reduced (3) | barrier artifact (confirmed) |
| GMRES `dot`/`axpy`/`scale` (`partial[]` reuse) | yes | yes | persists (62) | **barrier artifact (confirmed)** — see `reports/gmres_race_root_cause.md` |
| Interpolator cache (`point_data`/`hypercube_data`) | yes (heap-based) | yes (mixed) | **persists, heap-based (376)** | **confirmed real bug** |
| HYPRE/CPR/BoomerAMG internals | not reached | dominant (>1000), mostly barrier-shaped | dominant (>500), 60-85% barrier-shaped | **mostly barrier artifact — not confirmed as a bug** (see corrected Result 5) |
| `well_controls.cpp` / op-array reads | yes | — | persists (33) | same root cause as interpolator cache |

## Recommendation

One thing stands on solid, multi-tool evidence and is worth fixing:

1. **Add real synchronization to `multilinear_adaptive_cpu_interpolator`'s
   `point_data`/`hypercube_data` caches** (a lock, or a redesign that
   guarantees single-writer access) — confirmed by three independent
   tool/runtime combinations on real heap data, and unlike the HYPRE findings
   below, checked and found *not* to be barrier-boundary shaped.

The HYPRE/BoomerAMG question is **not settled either way** — most of what
was seen is explained by the same TSan+OpenMP barrier-annotation gap that
closed out the GMRES race, but a residual fraction wasn't individually ruled
out. If HYPRE's `nt>1` thread-safety actually matters for this project (per
`a5c8b1ac`'s GPU-pipeline concern), the way to get a real answer is a
LLVM `libomp` built *with* TSan support (`LIBOMP_TSAN_SUPPORT`/equivalent
CMake option) rather than the standard conda-forge package used here — that
would suppress the annotation-gap noise across the board and let whatever's
left (if anything) stand as genuine.

The GMRES `dot`/`axpy`/`scale` race is **closed, not a bug** —
`reports/gmres_race_root_cause.md` reproduced the identical false positive in
a ~15-line standalone program with no GMRES or engine code at all, on both
runtimes: neither a standard GCC `libgomp` nor a standard LLVM `libomp` build
annotates its team-join barrier for ThreadSanitizer, so *any* manual
reduction into a per-thread stack slot (a completely standard, correct
OpenMP idiom) gets flagged this way. No code change needed there.

Full raw log kept at (session scratchpad, not in the repo):
`tsan_report_clang.9134` (~40+ MB) / `tsan_clang_run_stdout.log`.
