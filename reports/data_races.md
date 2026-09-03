# ThreadSanitizer findings: `models/cpg_sloping_fault` (geothermal, `generate_5x3x4_wperiodic`, `OMP_NUM_THREADS=4`)

## How this was produced

- Build: `build_tsan/` — `CMAKE_BUILD_TYPE=RelWithDebInfo`, `OPENDARTS_CONFIG=MT`,
  `CMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer -g"`, GCC 15.2.
  (See `notes/gcc15_hypre.md` for a build issue hit and fixed along the way.)
- Run:
  ```
  LD_PRELOAD=$(gcc -print-file-name=libtsan.so) \
  OMP_NUM_THREADS=4 \
  TSAN_OPTIONS="halt_on_error=0:history_size=7:log_path=tsan_report" \
  python3 -c "from main import run; run(physics_type='geothermal', case='generate_5x3x4_wperiodic', out_dir='results_tsan_check', export_vtk=False)"
  ```
  (`LD_PRELOAD` is needed — without it, importing the sanitized `.so` fails with
  `cannot allocate memory in static TLS block`, a separate, unrelated TSan/glibc
  static-TLS issue with lazily `dlopen`'d instrumented libraries.)
- The run was killed after 5 minutes (TSan overhead is large; the case did not
  reach completion — it got through roughly 20 nonlinear iterations of a
  20-report-step case). Raw log: `tsan_report.2253` (~12 MB), 284
  `WARNING: ThreadSanitizer: data race` blocks.

**Correction to what I said in chat earlier:** I initially described several of
the categories below (op-array races in `assemble_jacobian_array`, the GMRES
vector-kernel races, the well-controls races) as straightforward concurrency
bugs in application code. After reading full stack pairs for representative
examples of each category (not just the top frame, which is what my first,
cruder pass at the log did), most of them turn out to have a specific, shared
shape that points somewhere else: **GCC's `libgomp` does not emit the runtime
annotations ThreadSanitizer needs to see its own barriers**, so TSan reports a
race for two accesses that are actually correctly ordered by an OpenMP
barrier in real execution. This is a documented category of TSan+GNU-OpenMP
false positive (Clang/LLVM's `libomp` is annotated for this; GCC's `libgomp`
is not). Only one category below is unambiguously a real, synchronization-free
bug independent of that question.

## Category 1 — real bug: the OBL interpolator cache has no locking, and IS accessed concurrently within a single active parallel region

**~250 of 284 races.** Files: `interpolation/include/multilinear_adaptive_cpu_interpolator.tpp`,
`interpolation/include/point_data_store.hpp`, `interpolation/include/multi_index_key.hpp`,
plus a long tail of `libstdc++` `std::_Hashtable` internals (`hashtable.h`,
`hashtable_policy.h`) that these two headers exercise.

`multilinear_adaptive_cpu_interpolator` keeps two `std::unordered_map` caches —
`point_data` (evaluated OBL points) and `hypercube_data` (assembled hypercube
payloads) — that are grown on demand ("adaptive": a cache miss triggers
evaluation and insertion, rather than everything being precomputed up front).
The class's own source comment documents the intended discipline:

```cpp
// interpolation/include/multilinear_adaptive_cpu_interpolator.tpp:257-260
// Phase 2c: assemble missing hypercube payloads in parallel from the now-complete
// point cache. The serial Phase 2b insertion loop above is the last point_data
// mutation before this OpenMP region; keep Phase 2c read-only unless point_data
// synchronization is introduced.
```

i.e. *within one call*, all mutation of `point_data` is meant to happen
single-threaded, strictly before the `#pragma omp parallel for` (line 266) that
reads it. That invariant is real and (as far as this run showed) not violated
within a single call.

The representative race I traced in full, though, is:

- **Read** — thread T6, inside `materialize_missing_cache(...) [clone ._omp_fn.0]`
  (a worker of the `#pragma omp parallel for` at `multilinear_adaptive_cpu_interpolator.tpp:266`),
  via `point_data_store::at()` → `std::unordered_map::find()` →
  `_Hashtable::_M_find_before_node` (`hashtable.h:2209`).
- **Previous write** — main thread, via `point_data_store::emplace()` →
  `std::unordered_map::emplace()` → `_Hashtable::_M_emplace_uniq` (`hashtable.h:2362`),
  which can trigger a **rehash** (reallocating the bucket array and re-chaining
  every node) — not just an append.
- Both point at the same heap-allocated hash-table storage (`Location is heap
  block of size 256`, i.e. individual hash nodes/bucket arrays, not stack).

The `emplace()` side is consistent with either (a) the serial Phase‑2b
insertion loop of a *different, later or earlier* call to
`interpolate_with_derivatives` (in which case this reduces to the same
"unannotated libgomp barrier" question as the other categories — see below),
or (b) `emplace()` being reached from more than one call site into the *same*
interpolator instance without any lock between them (a genuine bug). I did not
fully resolve which, because doing so requires walking every call site that
can trigger `materialize_missing_cache` for a given `operator_set_gradient_evaluator_iface*`
instance and confirming whether more than one of them can be live/interleaved
for the same interpolator object at once — out of scope for this pass.

What **is** clear regardless of that distinction: this cache has **no mutex,
no atomics, no `#pragma omp critical`, nothing** guarding `point_data` /
`hypercube_data` against concurrent mutation. It currently only works if the
call discipline described in the comment above is upheld by every caller,
across every interleaving of engine threads, for the lifetime of the
interpolator object — an invariant enforced by convention/comments only, not
by the type. Given `engine_super_cpu.tpp`'s own Jacobian-assembly loop also
calls into this same interpolator per grid block from multiple OpenMP threads
(see Category 2), and wells evaluate operators through it too, there are
multiple call paths into the same cache from what is, architecturally, a
multi-threaded engine. **This is the one finding I'd treat as an actual defect
to fix, independent of the libgomp/TSan question below** — either by giving
the cache a real lock (e.g. a `std::shared_mutex`, reader-lock for `at()`,
writer-lock for `emplace()`/`reserve()`), or by proving and enforcing
(not just documenting) that only one thread ever mutates it at a time.

## Category 2 — same shape, but on the `op_vals_arr` / `op_ders_arr` operator buffers

**~40 of 284 races.** `engines/src/engine_super_cpu.tpp` (`assemble_jacobian_array`,
many lines: 318, 327, 453, 481, 503, 577, 584, 590, 610, 633, 656, 665...)
reading `op_vals_arr[j * N_OPS + ...]` / `op_ders_arr[...]` for a neighbor
block `j`, racing against a write at
`interpolation/include/multilinear_interpolator_common.h:245/248`
(`interp_values[op] = workspace[op]; interp_derivs[...] = ...;` inside
`interpolate_point_with_derivatives_ws`, force-inlined into the interpolator's
own `interpolate_with_derivatives`/`materialize_missing_cache` machinery from
Category 1). `Location is heap block of size 84448`, i.e. the actual
`op_vals_arr` vector (`std::vector<value_t>`, resized once in
`engine_base::init_base`) — not a stack temporary.

This is the same underlying data path as Category 1 (both are the adaptive
interpolator filling in `op_vals_arr`/its own caches for a given block), just
observed from the Jacobian-assembly side rather than the interpolator side.
Whatever the resolution of Category 1 turns out to be (real missing lock vs.
a barrier TSan can't see), fixing Category 1 will very likely collapse most
or all of Category 2 with it, since they're the same handful of shared
buffers seen from two call sites.

## Category 3 — likely false positives: accesses that should be separated by an (unannotated) OpenMP barrier

**The remainder (well_controls.cpp races, the GMRES `dot`/`axpy`/`scale` races,
`engine_base::init_jacobian_structure`, and the `engine_super_cpu.tpp:153`
self-race).** Representative, fully-traced examples:

- `engine_base.cpp:99` (serial code, right after a `#pragma omp parallel`
  block closes) racing against `numa_set`/`memset` at `engine_base.cpp:82`,
  attributed to a worker thread (`[clone ._omp_fn.0]`) that — per the OpenMP
  standard — **must** have finished before the closing `}` of that parallel
  region lets the master thread continue. Same shape recurs for the *next*
  parallel region in the same function (`engine_base.cpp:119` vs. `:71`, the
  pragma lines of the second and first `#pragma omp parallel` blocks in
  `init_jacobian_structure`).
- `engine_super_cpu.tpp:153` (`int id = omp_get_thread_num();`, the very first
  line of `assemble_jacobian_array`'s parallel region) raced against *itself*
  — but the two accesses are from **two different, unrelated calls**: one from
  `engine.init()` (thread T6 created there) and one from a later
  `assemble_linear_system()` call, on the *same reused stack address*
  (`Location is stack of main thread`). This is a stack-slot-reuse-across-calls
  pattern, not two threads genuinely touching the same logical variable.
- GMRES's `dot()` reduction buffer (`linsolv_gmres.cpp:180` init loop vs.
  `:187` per-thread store into `partial[omp_get_thread_num()]`) — also
  `Location is stack of main thread`, and again a stale write from a
  thread (T5) whose owning parallel region, in a correctly-synchronized
  execution, is fully joined before the next `dot()` call reuses that stack
  slot.
- `well_controls.cpp:321`'s write into the Jacobian racing against a `memset`
  — this is the `numa_set(Jac, 0, ...)` first-touch zeroing at the top of
  `assemble_jacobian_array` (`engine_super_cpu.tpp:159`), i.e. the same
  region-boundary shape as the first bullet, just with the Jacobian buffer
  (`Location is heap block of size 33600`) instead of a stack array — heap vs.
  stack doesn't change the argument, only whether TSan's summary line happens
  to say "stack of main thread" or names an allocation site.

In every one of these traced cases, the two racing accesses are steps that
are supposed to be strictly ordered by a preceding `#pragma omp parallel`
region's implicit barrier (or, in the `dot`/`id` cases, by the fact that the
previous parallel region must have fully joined before the enclosing function
returned). GCC's `libgomp` barrier implementation does not carry the
`AnnotateHappensBefore`/`AnnotateHappensAfter`-equivalent hooks ThreadSanitizer
needs to see that ordering (this is a known gap relative to LLVM's `libomp`,
which is built with explicit TSan annotations for exactly this reason). TSan's
own history-based race detector then reports "no happens-before edge found
between these two accesses" — correctly, from what it can observe — even
though the real GOMP runtime does order them correctly via a barrier it
cannot see into.

**I have not independently confirmed this with a second, TSan-annotated
OpenMP runtime** (e.g. rebuilding with Clang + LLVM's `libomp` and re-running
the same case) — that would be the definitive way to separate genuine bugs
from this artifact class, since `libomp`'s barriers are TSan-visible and
would not reproduce these particular reports if they are indeed artifacts.
That's the natural next step before spending effort "fixing" anything in this
category.

## Practical takeaway

| Category | Count (of 284) | Confidence it's a real bug |
|---|---|---|
| 1. OBL interpolator cache (`point_data`/`hypercube_data`) unguarded | ~250 | High that there's *no lock*; medium-high that it's *exercised* concurrently rather than only racing across an unannotated barrier — needs the call-site audit described above |
| 2. `op_vals_arr`/`op_ders_arr` read/write in `assemble_jacobian_array` | ~40 | Same as (1); it's the same underlying cache seen from the consumer side |
| 3. `init_jacobian_structure`, `assemble_jacobian_array` self-race at `:153`, GMRES `dot`/`axpy`/`scale`, `well_controls.cpp` vs. `numa_set` | remainder | Low — all traced examples match the "GCC libgomp barrier invisible to TSan" shape; recommend re-testing under Clang+libomp before treating as defects |

**Recommended next step, in order:**
1. Re-run the identical case under a Clang + `libomp` ThreadSanitizer build.
   Whatever disappears is Category 3 (artifact); whatever remains is real.
2. Independent of (1): add real synchronization to the interpolator's
   `point_data`/`hypercube_data` maps (Category 1/2) — a lock is a strict
   improvement whether or not the current races are "genuinely concurrent" or
   "barrier-invisible," since it removes the reliance on an implicit,
   comment-documented-only ordering contract.
3. Let the case run to completion (this pass only covered the first ~20
   nonlinear iterations before the 5-minute cutoff) for a fuller picture,
   once (1) narrows down what's worth looking at.

Full raw log kept at (session scratchpad, not in the repo):
`tsan_report.2253` / `run_stdout.log`.
