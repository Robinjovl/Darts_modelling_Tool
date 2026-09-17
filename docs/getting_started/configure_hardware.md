# Configure hardware usage

## Multi-thread with openMP

To set the number of threads (CPU cores) to be used, you must have had compiled open-darts with the multi-thread option (`OPENDARTS_CONFIG=MT`, or a GPU build, which enables host-side OpenMP as well; the default is the single-thread `ST` configuration).

In such a build OpenMP covers the in-tree `open-darts/linear_solvers` too: the iterative solvers (FGMRES, CPR/FS-CPR and the Schur elimination) run multi-threaded, and HYPRE contributes its own OpenMP parallelism when it was built with `HYPRE_ENABLE_OPENMP=ON`. The direct solvers are the exception -- SuperLU is not parallelized ([issue 40](https://gitlab.com/open-darts/open-darts/-/issues/40)).

Then add to your python script:

```python
from darts.engines import set_num_threads
set_num_threads(NT)
```

Half of the cores available are used unless specified via `set_num_threads` or via setting the environment variable `export OMP_NUM_THREADS=NT`.

<div class="warning">

If the number of threads requested `NT` is larger than the available you might get a Segmentation fault or a BUS error.

</div>

<div class="warning">

Results depend on the thread count. At a *fixed* `NT` a run is reproducible, but changing `NT` groups the
parallel reductions differently, so the linear solver returns answers that differ at ULP level and the
Newton path can amplify that. Pin the thread count whenever you compare against reference results -- the
test suite runs with `OMP_NUM_THREADS=1` for exactly this reason.

</div>

## GPU

Turn on GPU usage in calculation by adding the next lines at the start of the python script:
Add `platform='gpu'` at `init` call, for example:

```python
model.init(platform='gpu')
```

If you would like to change the GPU device, add these lines to your model script:

```python
from darts.engines import set_gpu_device
set_gpu_device(N)
...
m.init(platform='gpu')
```

with `N` being your GPU device number. For example if you have 2 GPUs, you can call `set_gpu_device(0)` or `set_gpu_device(1)`.
The function set_gpu_device` should be called before `Model.init()`.

Only NVIDIA GPUs are currently supported. It is possible to use one GPU for one simulation.
