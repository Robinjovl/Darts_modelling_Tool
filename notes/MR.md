# Data race investigation: interpolator cache fix + build portability fixes

## Summary

Investigated data races in `models/cpg_sloping_fault` (geothermal,
`generate_5x3x4_wperiodic`, `OMP_NUM_THREADS=4`) using ThreadSanitizer (GCC
+ `libgomp`, and Clang + LLVM `libomp`) and Helgrind. Found and fixed one
confirmed, real thread-safety bug in the adaptive OBL interpolator's cache.
Along the way, fixed four unrelated, previously-latent build bugs that
surfaced only because this required building with GCC 15 and, for the first
time, Clang on Linux.

Full investigation write-ups are in `reports/`:
- `reports/data_races.md` — initial GCC + `libgomp` ThreadSanitizer pass, findings and categorization.
- `reports/data_races_helgrind.md` — cross-check with Helgrind.
- `reports/data_races_clang_libomp.md` — cross-check with Clang + LLVM `libomp`.
- `reports/gmres_race_root_cause.md` — root-causes the GMRES vector-kernel
  "race" as a proven ThreadSanitizer+OpenMP limitation, not a bug (minimal
  standalone repro included).
- `reports/gcc15_hypre.md` — the GCC 15 / HYPRE `omp.h` build failure hit along the way.

## The fix: interpolator cache locking

`interpolation/include/multilinear_adaptive_cpu_interpolator.{hpp,tpp}`

`multilinear_adaptive_cpu_interpolator`'s `point_data` and `hypercube_data`
caches had no synchronization at all, despite being reachable from more than
one call path with no coordination between them: the batched
`interpolate_with_derivatives()`/`materialize_missing_cache()` path (which
only protects itself via internal phase ordering — serial materialize, then
parallel read) and the single-point `get_point_data()`/`get_hypercube_data()`
path (used e.g. for well-block evaluation), which can run concurrently with
the batched path or with itself. Confirmed as a genuine bug (not a sanitizer
artifact) across three independent tool/runtime combinations, each showing
raw `std::unordered_map` internal races (rehash vs. concurrent lookup,
concurrent `emplace`) on real heap-allocated storage.

Fix: two `std::shared_mutex`es (`point_data_mutex_`, `hypercube_data_mutex_`)
guarding every read/write path — shared (reader) locks for lookups, so the
hot per-cell/per-hypercube read loops still run fully concurrently with each
other; exclusive locks only around actual cache misses (insertion) and
eviction. `hypercube_data_mutex_` is only ever acquired after
`point_data_mutex_` in the same call, never the reverse, so the two can't
deadlock.

Also fixed along the way: `get_point_data()`'s cache-miss path was writing
into `this->new_point_coords`/`this->new_operator_values` — per-object
scratch buffers on the base class — instead of local variables. Since this
function is reachable from multiple threads, that was a second, independent
race on top of the map itself; switched to locals.

Verified with a pre/post ThreadSanitizer comparison on the same case: the
interpolator-cache-specific race reports (previously the majority of all
reports in a similarly-sized run) dropped to near zero after the fix, with
the simulation still converging normally (no deadlock, no behavior change).

## Build portability fixes (unrelated to the race investigation itself)

Needed to get GCC 15 and, for the first time, Clang-on-Linux builds working
at all — none of these are about threading:

1. **`linear_solvers/include/Types.hpp`** — removed a redundant outer
   `extern "C" { #include <_hypre_utilities.h> }` wrap. That header already
   self-guards internally; the outer wrap also pulled HYPRE's own
   `#include <omp.h>` (which precedes HYPRE's internal guard) into C linkage,
   and GCC 15's `omp.h` declares C++ template overloads there — illegal
   under `extern "C"` ("template with C linkage"). See `reports/gcc15_hypre.md`.
2. **`CMakeLists.txt`** — the Clang compiler-flags branch unconditionally
   passed `-Wl,-no_compact_unwind`, an Apple `ld64`-only linker flag; GNU
   `ld`/`lld` (used by Clang on Linux) reject it outright. Gated to
   `if(APPLE)`.
3. **`discretizer/src/approximation.h`** — `constexpr index_t nb1 = ap1.n_block;`
   accessed a static constexpr member through a dependent-type function
   parameter; Clang rejected it as non-constant. Switched to the
   fully-qualified form (`LinearApproximation<VarNames1...>::n_block`).
4. **`discretizer/src/mech_discretizer.cpp`** — two issues:
   - An explicit template instantiation (`template class MechDiscretizer<POROELASTIC>;`)
     appeared after only `using namespace dis;`, not inside `namespace dis`.
     Per the standard, explicit instantiation of a namespace member must
     occur in that namespace; a `using` directive doesn't count. GCC accepts
     this as an extension, Clang correctly rejects it. Wrapped in
     `namespace dis { ... }`.
   - Once that compiled, a static member initializer
     (`N_UNKNOWNS.at(MODE)`, reading a *different* static map) crashed on
     import under Clang with `unordered_map::at: out_of_range` — a genuine
     static-initialization-order-fiasco that GCC's particular link order
     happened to get right by luck. Fixed by computing the value directly
     instead of depending on another static object's construction order.

## Testing performed

- TSan (GCC 15 + `libgomp`), TSan (Clang 23 + LLVM `libomp`, built from
  scratch including HYPRE/SuperLU under the same toolchain), and Helgrind —
  all on `models/cpg_sloping_fault`, geothermal, `generate_5x3x4_wperiodic`,
  `OMP_NUM_THREADS=4`.
- Pre-fix and post-fix TSan runs compared directly to confirm the
  interpolator-cache race category is resolved.
- No functional/numerical regression observed (simulation converges the
  same way before and after the lock fix).

## Known open items (not fixed, documented instead)

- **GMRES `dot`/`axpy`/`scale` "races"** (`linear_solvers/src/linsolv_gmres.cpp`):
  proven to be a ThreadSanitizer+OpenMP limitation via a minimal,
  from-scratch reproduction — neither a standard GCC `libgomp` nor a
  standard LLVM `libomp` build annotates its team-join barrier for
  ThreadSanitizer, so any manual reduction into a per-thread stack slot
  (a completely standard OpenMP idiom) gets flagged this way. No code
  change made or needed. See `reports/gmres_race_root_cause.md`.
- **HYPRE/BoomerAMG internal "races"**: mostly the same barrier-artifact
  shape as above (confirmed for 60-85% of sampled examples across several
  categories), but not fully ruled out for the remainder. Not fixed here.
  If HYPRE's `nt>1` thread-safety needs a real answer, the next step is
  testing against an LLVM `libomp` built *with* TSan support enabled
  (non-default build option), which would suppress this artifact class
  entirely and let anything still standing be trusted as genuine. See
  `reports/data_races_clang_libomp.md` (Result 5, corrected).
