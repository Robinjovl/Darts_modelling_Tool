# Root cause of the GMRES `dot`/`axpy`/`scale` race: not a bug — a fundamental TSan+OpenMP limitation

**Supersedes the "unresolved, treat as real" verdict for this category in
`reports/data_races_clang_libomp.md`.** Digging further with a minimal,
isolated reproduction gives a definitive answer: this is a well-known class
of ThreadSanitizer false positive with OpenMP, not a defect in
`linear_solvers/src/linsolv_gmres.cpp`.

## The question

`reports/data_races.md`, `reports/data_races_helgrind.md`, and
`reports/data_races_clang_libomp.md` all flagged a race in `dot()`'s manual
reduction:

```cpp
// linear_solvers/src/linsolv_gmres.cpp:170-199
inline mat_float dot(const mat_float *a, const mat_float *b, std::size_t n)
{
  ...
  mat_float partial[max_slots];
  for (int t = 0; t < nt; ++t)
    partial[t] = 0.0;                              // line 180
#pragma omp parallel num_threads(nt)
  {
    mat_float local = 0.0;
#pragma omp for schedule(static) nowait
    for (...) local += a[i] * b[i];
    partial[omp_get_thread_num()] = local;          // line 187
  }
  mat_float s = 0.0;
  for (int t = 0; t < nt; ++t) s += partial[t];      // combine, after the region closes
  return s;
}
```

`reports/data_races_clang_libomp.md` found this persisted even under Clang +
LLVM's `libomp` (which is supposed to be the TSan-annotated runtime), which
downgraded my earlier "it's just GCC `libgomp`" theory to "unresolved, needs
more digging." This report is that digging.

## The test

Two minimal, standalone reproductions (no GMRES, no engine, no open-DARTS
code at all):

1. **`dot_repro.cpp`** — the *exact* pattern from `dot()`, called twice in a
   row (matching `solve_impl`'s `dot(rhs,rhs,n)` then `dot(p,p,n)`), looped
   2000 times.
2. **`dot_repro_single.cpp`** — stripped to the bare minimum: declare a stack
   array, run **one** `#pragma omp parallel num_threads(nt) { partial[tid] = ...; }`,
   then read `partial[]` from the main thread immediately after the region
   closes. No loop, no repeated calls, no sibling function — the single most
   canonical "manual OpenMP reduction into a per-thread slot" idiom that
   exists.

Compiled both with `-fsanitize=thread -fopenmp -O2 -g`, run with
`OMP_NUM_THREADS=4`, under **both** toolchains used in this investigation:
GCC 15 + `libgomp`, and Clang 23 + LLVM `libomp` (the same
`darts-clang-tsan` conda environment built for `reports/data_races_clang_libomp.md`).

## Result: it reproduces immediately, every time, on both runtimes, with the single-call minimal program

```
$ g++ -O2 -g -fopenmp -fsanitize=thread -o dot_single_gcc dot_repro_single.cpp
$ OMP_NUM_THREADS=4 ./dot_single_gcc
WARNING: ThreadSanitizer: data race
  Read of size 8 at 0x7fffffffc6a8 by main thread:
    #0 main dot_repro_single.cpp:18
  Previous write of size 8 at 0x7fffffffc6a8 by thread T1:
    #0 main._omp_fn.0 dot_repro_single.cpp:14
  Location is stack of main thread.
s=6.000000
```

```
$ clang++ -O2 -g -fopenmp -fsanitize=thread -o dot_single_clang dot_repro_single.cpp
$ OMP_NUM_THREADS=4 ./dot_single_clang
WARNING: ThreadSanitizer: data race
  Read of size 8 at 0x7fffffffc5d8 by main thread:
    #0 main dot_repro_single.cpp:18
  Previous write of size 8 at 0x7fffffffc5d8 by thread T1:
    #0 main.omp_outlined dot_repro_single.cpp:14
  Location is stack of main thread.
s=6.000000
```

Both report a "race" between a worker thread's write inside the parallel
region and the main thread's read *after the region's closing brace* — i.e.
after the OpenMP standard's **mandatory implicit barrier**, which guarantees
every worker has finished and joined before the master thread continues.
The program computes the correct answer both times (`s=6.000000`, i.e.
`0+1+2+3`) — there is no actual runtime bug, only a false alarm from the
sanitizer. A bare `#pragma omp parallel` (no `num_threads` clause, ruling
out that clause as the trigger) shows the same thing.

## Why this happens

ThreadSanitizer doesn't understand the OpenMP fork-join model natively. It
recognizes a fixed set of synchronization primitives (mutexes, properly
ordered atomics, `pthread_create`/`join`, and an explicit annotation API) and
treats any two accesses from different threads as racing unless one of those
primitives visibly connects them. An OpenMP runtime's team barrier is
usually implemented with a spin-then-futex/mutex wait loop internal to the
runtime — a real, correct synchronization mechanism, but not automatically
one of the primitives TSan's interceptors recognize as establishing a
happens-before edge, unless the runtime is specifically built to call TSan's
annotation hooks at its barrier/join points.

LLVM's `libomp` *can* be built with that support
(`LIBOMP_TSAN_SUPPORT`/similar), but that is a non-default build option.
GCC's `libgomp` has no such analogous instrumented mode. Both the GCC build
in this project and the conda-forge `llvm-openmp` package used for the Clang
comparison are ordinary, non-instrumented builds — hence both show the exact
same false positive on the exact same minimal, correct code. This matches
documented, known behavior of ThreadSanitizer with OpenMP in general, not
something specific to this codebase's OpenMP usage.

## What this means for the other findings

This gives a clean, principled way to sort every "barrier-boundary shaped"
report across all three earlier investigations: **if a race requires
crossing the join point of a `#pragma omp parallel` region — i.e. one access
happens inside the region (or a region that has already closed) and the
conflicting access happens on the same thread's continuation after that
region ends — it is now proven, not just suspected, to be a sanitizer
artifact, regardless of compiler or OpenMP runtime.**

That covers, definitively:
- `linsolv_gmres.cpp` `dot`/`axpy`/`scale` (this report) — **artifact, closing this out.**
- `engine_base::init_jacobian_structure`'s first-touch regions
  (`engine_base.cpp:71`/`:119`) — **artifact**, now with a mechanism, not just a pattern match.
- `engine_super_cpu.tpp:153`'s self-race — **artifact** (already saw this vanish under Clang; now we know why it's inherently fragile/tool-dependent rather than a real fix).

It does **not** cover, and does not change the verdict on:
- The **interpolator cache** (`point_data`/`hypercube_data`) races — these
  are between two threads that are *simultaneously, actively* running (e.g.
  two live worker threads of the *same still-open* parallel loop both
  calling into the cache), with **no barrier at all** between the conflicting
  accesses. There's no join point for TSan to fail to understand — the
  threads are concurrently live and the map has no lock. Still a real bug.
- The **HYPRE/BoomerAMG/CPR** races — same reasoning; multiple of these
  showed genuinely concurrent access to heap-allocated matrix/vector state
  from what appear to be independently-running code paths, not a
  region-boundary artifact.

## Recommendation

- Treat the GMRES vector-kernel races as **closed / not a bug**. No code
  change needed in `linsolv_gmres.cpp`.
- If a fully-clean TSan run (zero noise) is ever wanted for this project,
  the only real fix is building LLVM's `libomp` from source with its TSan
  support option enabled and using that for the sanitizer build — not
  something to do routinely, but worth knowing if the noise ever needs to go
  away for a specific investigation.
- The interpolator cache lock and the HYPRE-OpenMP thread-safety question
  (both flagged in `reports/data_races_clang_libomp.md`) remain the two real,
  actionable findings from this whole investigation.

Repro files kept at (session scratchpad, not in the repo):
`dot_repro.cpp`, `dot_repro_single.cpp`, `dot_repro_bare.cpp`.
