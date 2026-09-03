# Build failure: `template with C linkage` in `omp.h` (GCC 15 + HYPRE)

## Symptom

Building `linear_solvers` (specifically the MGR sources — `mgrStrategy.cpp`,
`mgr_linear_solver.cpp`, `mgr_compositional_flow_strategy.cpp`) failed with
dozens of errors of this shape:

```
/usr/lib/gcc/x86_64-linux-gnu/15/include/omp.h:486:1: error: template with C linkage
  486 | template<typename __T, typename __U, omp_allocator_handle_t __Handle>
      | ^~~~~~~~
/home/ilshat/projects/open-darts_solvers/linear_solvers/src/../include/Types.hpp:18:1: note: ‘extern "C"’ linkage started here
   18 | extern "C" {
      | ^~~~~~~~~~
```

repeated for every C++ template declared in `omp.h` (the `__allocator_templ`
helper templates and the `operator==`/`operator!=` overloads it defines for
`omp_alloc`/`omp_free`-based custom allocators).

This is a genuine compile error, not a sanitizer artifact — it reproduced on
a plain (non-TSAN) rebuild too, it just happened not to be hit yet because
the affected `.o` files were already built and cached from before GCC was
upgraded to 15.

## Root cause

`linear_solvers/include/Types.hpp` pulls in HYPRE's C API like this
(before the fix):

```cpp
// HYPRE types
extern "C" {
#include <_hypre_utilities.h>
}
```

The intent is reasonable — HYPRE is a C library, so its declarations need
`extern "C"` linkage when included from C++. The problem is that
`_hypre_utilities.h` **already does this itself**, and it does it
*selectively*:

```c
/* thirdparty/install/include/_hypre_utilities.h */
#include "HYPRE_utilities.h"

#ifdef HYPRE_USING_OPENMP
#include <omp.h>          // <-- included BEFORE the extern "C" guard below
#endif
...
#ifdef __cplusplus
extern "C" {              // <-- HYPRE's own guard starts here
#endif
```

Because this build has `HYPRE_USING_OPENMP` defined (open-DARTS builds HYPRE
with `HYPRE_ENABLE_OPENMP=ON` by default — see `helper_scripts/build_darts_cmake.sh`),
`_hypre_utilities.h` includes `<omp.h>` *before* it opens its own
`extern "C"` block. HYPRE's authors clearly intended `omp.h` to be included
as ordinary C++ (its C++-only content should get normal C++ linkage), with
only HYPRE's *own* C declarations that follow going into `extern "C"`.

`Types.hpp`'s outer `extern "C" { #include <_hypre_utilities.h> }` defeats
that separation: the *entire* contents of `_hypre_utilities.h`, including its
`#include <omp.h>` line, are now textually inside `Types.hpp`'s `extern "C"`
block. The preprocessor doesn't care about linkage specifiers when expanding
`#include`, so `omp.h`'s full contents — HYPRE's nested `extern "C"` included
— end up nested *inside* `Types.hpp`'s outer `extern "C"`. Nesting
`extern "C"` blocks is normally harmless (it's idempotent for plain
declarations), but it means *everything* in `omp.h`, including the parts
HYPRE's header deliberately left outside any `extern "C"`, now has C linkage
forced onto it.

Since GCC 13 (and still in the current GCC 15.2 on this machine), libgomp's
`omp.h` has grown a genuine C++-only section: template-based `omp_alloc`
wrappers (`std::__allocator_templ<T, Handle>` and friends,
`operator==`/`operator!=` overloads for them) used to implement the
`omp::default_mem` / `omp::large_cap_mem` / etc. C++ allocator helpers added
to the OpenMP 5.x C++ API. Templates are a C++-only language construct with
no meaning in C, so the C++ standard flatly disallows declaring a template
under `extern "C"` linkage — a template's linkage is a C++-specific concept
that `extern "C"` cannot express, regardless of whether the surrounding code
is actually compiled as C++. GCC diagnoses this as a hard error
(`template with C linkage`) rather than silently ignoring the linkage
specifier.

So the failure chain is:

1. HYPRE built with OpenMP support (`HYPRE_ENABLE_OPENMP=ON`, the default).
2. `_hypre_utilities.h` therefore `#include`s `<omp.h>`, intentionally
   *before* its own `extern "C"` guard.
3. `Types.hpp` wraps the whole header include in an extra, redundant
   `extern "C" { ... }`.
4. That outer wrap also captures `omp.h`'s contents.
5. GCC 15's `omp.h` contains C++ templates in that captured region.
6. `template with C linkage` — hard compile error, once per template
   declaration encountered (hence the long repeated error list, one line per
   template/operator overload in `omp.h`).

This is why the error only started appearing now: it depends on the
combination of (a) HYPRE being built with OpenMP enabled and (b) a libgomp
`omp.h` new enough to declare the C++ allocator templates. Neither
`Types.hpp`'s pattern nor HYPRE's header changed recently in this repo — the
trigger was the toolchain moving to GCC 15.

## Why other files in the codebase didn't hit this

Every other translation unit that needs `_hypre_utilities.h` (e.g.
`linsolv_hypre_amg.cpp`, `linsolv_hypre_ilu.cpp`) includes it *without* an
extra `extern "C"` wrapper:

```cpp
// engines/.../linsolv_hypre_amg.cpp
#include "_hypre_utilities.h"
#include "HYPRE.h"
...
```

They rely on the header's own internal `#ifdef __cplusplus` guard, which is
exactly what it's designed for. `Types.hpp` (added as part of the newer MGR
solver work on this branch) was the only call site that added a second,
redundant layer — and the only one that broke.

## Fix

Remove the outer `extern "C"` wrapper in `Types.hpp` and let
`_hypre_utilities.h` guard itself, matching the pattern already used
elsewhere in the codebase:

```cpp
// HYPRE types
// Note: _hypre_utilities.h already guards its own declarations with
// `#ifdef __cplusplus / extern "C"` internally. Wrapping the #include itself
// in an outer extern "C" block also pulls in HYPRE's own `#include <omp.h>`
// (which precedes HYPRE's internal guard) under C linkage. Newer libgomp
// omp.h (GCC 15+) declares C++ template overloads there, which is illegal
// under extern "C" ("template with C linkage").
#include <_hypre_utilities.h>
```

File: `linear_solvers/include/Types.hpp` (diff, currently uncommitted):

```diff
 // HYPRE types
-extern "C" {
+// Note: _hypre_utilities.h already guards its own declarations with
+// `#ifdef __cplusplus / extern "C"` internally. Wrapping the #include itself
+// in an outer extern "C" block also pulls in HYPRE's own `#include <omp.h>`
+// (which precedes HYPRE's internal guard) under C linkage. Newer libgomp
+// omp.h (GCC 15+) declares C++ template overloads there, which is illegal
+// under extern "C" ("template with C linkage").
 #include <_hypre_utilities.h>
-}
 
 namespace mgr {
```

No functional behavior changes: the C declarations HYPRE cares about still
get `extern "C"` linkage, just from HYPRE's own guard rather than a redundant
outer one, and the `omp.h` content in between is left with correct (normal
C++) linkage.

## General lesson

Never wrap a third-party header's `#include` in your own `extern "C" { }`
block unless you've checked that the header doesn't itself pull in anything
that needs to stay outside `extern "C"` (system headers, other C++-aware
headers pulled in transitively, etc.). If the header already self-guards
with `#ifdef __cplusplus`/`extern "C"` — as essentially every well-behaved C
library header does — just `#include` it plainly and trust its own guard.
