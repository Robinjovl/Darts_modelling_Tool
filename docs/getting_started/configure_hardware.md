# Configure hardware usage

## Multi-thread with openMP

To set the number of threads (CPU cores) to be used, you must have had compiled open-darts with the multi-thread option. Also, `open-darts/solvers` do not support multi-thread with openMP yet.

Then add to your python script:

```python
from darts.engines import set_num_threads
set_num_threads(NT) 
```

Half of the cores available are used unless specified via `set_num_threads` or via setting the environment variable `export OMP_NUM_THREADS=NT`.

<div class="warning">

If the number of threads requested `NT` is larger than the available you might get a Segmentation fault or a BUS error.

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
```

with `N` being your GPU device number. For example if you have 2 GPUs, you can call `set_gpu_device(0)` or `set_gpu_device(1)`.
