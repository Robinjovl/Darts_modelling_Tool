# Parallel operator interpolation and evaluation

This page explains the design introduced to parallelise the **adaptive OBL operator
update** and how to use it in a model. Two independent layers of concurrency are
involved; they are configured separately and should be understood together.

| Layer | Where | Knob | Parallelises |
| --- | --- | --- | --- |
| C++ OpenMP | `multilinear_adaptive_cpu_interpolator`, `linear_adaptive_cpu_interpolator` | `OMP_NUM_THREADS` | cache lookup, hypercube assembly, interpolation |
| Python multiprocessing | `ParallelEvaluator` | `n_workers` | evaluation of *missing* supporting points (flash / properties) |

The two are orthogonal: the C++ layer accelerates interpolation arithmetic that is
already pure C++; the Python layer accelerates supporting-point generation, which
calls back into Python physics and therefore cannot run under OpenMP threads (GIL,
non-thread-safe flash solvers).

## 1. C++ interpolator — three-phase adaptive update

The adaptive interpolators previously generated missing points/hypercubes inside an
`omp single` region and then interpolated. The update splits
`interpolate_with_derivatives` into three explicit phases so that the only serial
part is the unavoidable one:

* **Phase 1 — discover (parallel).** For every requested cell the hypercube index is
  computed from read-only axis parameters (`omp parallel for`). Unique indices are
  collected and filtered against `hypercube_data` to find what is missing.
* **Phase 2 — materialise (mixed).** Missing *supporting points* are gathered, packed
  into one flat coordinate array and evaluated through a **single**
  `evaluate_batch()` call (one GIL acquire, one Python round-trip). Results are
  committed serially into `point_data`. Missing *hypercube payloads* are then
  assembled in parallel into a dense scratch vector and committed serially into
  `hypercube_data`.
* **Phase 3 — interpolate (parallel).** With `point_data`/`hypercube_data` now
  read-only, every cell is interpolated concurrently. Each thread keeps a reusable
  workspace (`interpolate_point_with_derivatives_ws`), so the hot loop performs no
  heap allocation.

After Phase 2 the shared maps are never mutated again, which is what makes Phase 3
safe to run without locks. Because every phase is either serial or operates on
independent per-cell / per-hypercube data, the interpolated result is
**bit-identical regardless of `OMP_NUM_THREADS`** (verified by
`tests/interpolators/test_interpolation.py::test_interpolator_thread_consistency`).
The linear adaptive interpolator follows the same idea with a lighter two-pass form
(`materialize_missing_points` + base interpolation).

This layer needs **no model changes** — any model using an adaptive `multilinear`
or `linear` interpolator benefits automatically once built with OpenMP
(`OPENDARTS_CONFIG=MT`) and run with `OMP_NUM_THREADS > 1`.

## 2. Python `ParallelEvaluator` — parallel supporting-point evaluation

`darts.physics.base.parallel_evaluator.ParallelEvaluator` wraps an evaluator with a
persistent `multiprocessing.Pool`. It adds `evaluate_batch(states, n_points, values,
n_ops)`, which the C++ Phase 2 calls; each worker process holds its own independent
evaluator, giving true parallelism for CPU-bound, non-thread-safe flash/property code.

`evaluate_batch` is a new, non-pure virtual on `operator_set_evaluator_iface` with a
serial C++ default, so existing evaluators keep working unchanged. `OperatorsBase`
also provides a serial Python `evaluate_batch`.

`evaluate()` (single point) always uses a local serial evaluator; the pool is used
only for batched supporting-point materialisation. Small batches
(`n_points <= n_workers`) skip the pool, and any pool error falls back to serial
evaluation with a warning rather than aborting the run.

## 3. Using it in a model

Both layers are enabled from `Model.init`:

```python
m.init(itor_type='multilinear', platform='cpu',
       parallel_evaluation=True, n_workers=8)
```

```bash
export OMP_NUM_THREADS=8     # C++ interpolation threads
```

`parallel_evaluation=True` wraps each `reservoir_operators[region]` with a
`ParallelEvaluator`. The evaluator for each worker is produced by a *factory* —
and that factory is **derived automatically from the model**, with no per-model
boilerplate.

### The default factory — `ModelEvaluatorFactory`

`DartsModel.get_evaluator_factory(region)` returns a
`ModelEvaluatorFactory` by default. It is a small, picklable object that stores the
model class and the constructor arguments the model was created with (captured in
`DartsModel.__new__`). When a worker process calls it, it:

1. reconstructs the model — running the model's own `set_reservoir`/`set_physics`;
2. calls `physics.set_operators()` to build the operator objects;
3. returns `physics.reservoir_operators[region]`.

This **reuses the model's own `set_physics`/`PropertyContainer` construction**, so
there is no duplication of the property/flash/kinetics stack and the mechanism works
for any model whose constructor arguments are picklable. `model.init()` is *not*
called in the worker — no engine and no nested worker pool are created.

A model therefore needs **nothing beyond** passing `parallel_evaluation=True`.
`Model` subclasses do not implement `get_evaluator_factory`; see
`models/chemistry/carbonated_water` for a working example that relies entirely on
the default.

### When to override `get_evaluator_factory`

Override it only if:

* model reconstruction is too expensive to repeat per worker (e.g. a large
  unstructured reservoir with heavy meshing — the workers would rebuild it), or
* the model's constructor arguments are not picklable.

An override must return a **picklable** callable `() -> operator_set_evaluator_iface`
(a top-level callable or another `ModelEvaluatorFactory`-like class — never a nested
function / lambda / closure). `ParallelEvaluator` checks picklability eagerly and
raises a clear error otherwise.

## 4. Windows / macOS support

The factory must be picklable because, under the `spawn` multiprocessing start
method — the default on **Windows and macOS** — the pool pickles the factory to ship
it to each worker (under Linux `fork` it is inherited instead). `ModelEvaluatorFactory`
is a plain top-level class holding only the model class and plain-data constructor
arguments, so it pickles correctly under both `fork` and `spawn`. The worker
functions are likewise module-level.

`ParallelEvaluator` accepts an optional `start_method` argument; the test suite uses
`start_method='spawn'` to exercise the Windows/macOS path on any platform
(`test_interpolation.py::test_parallel_evaluator`).

## 5. Status across CI/CD models

* The **C++ OpenMP layer** is in effect for every CI/CD model that uses an adaptive
  interpolator — no integration work, only an `MT` build.
* The **`ParallelEvaluator` layer** is opt-in per model via `parallel_evaluation=True`.
  Thanks to the default `ModelEvaluatorFactory` it requires no model-specific code,
  so any model with picklable constructor arguments can enable it. It is currently
  exercised by `models/chemistry/carbonated_water`.

## 6. Caveats

* Each worker (and the parent) reconstructs the model once at pool start-up. For
  models with expensive reservoir construction this is a real cost — override
  `get_evaluator_factory` with a lighter factory in that case.
* `OMP_NUM_THREADS` and `n_workers` are independent. Over-subscription thrashes the
  CPU; a reasonable starting point is `n_workers × OMP_NUM_THREADS ≲ physical cores`.
* NaN supporting points are warned about but still cached; a failed flash silently
  degrades accuracy rather than stopping the run.
* Only `reservoir_operators` are wrapped — property and well operators keep the
  serial path.
