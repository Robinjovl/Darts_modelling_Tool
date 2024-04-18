# Configure hardware usage

To set the number of threads (CPU cores) to be used, add:

```python
from darts.engines import set_num_threads
set_num_threads(NT) 
```

All cores available are used by default.

Turn on GPU usage in calculation by adding the next lines at the start of the python script:
Add `palatform='gpu'` at `set_physics` call, for example:

```python
super.set_physics(physics, platform='gpu')
```

If you would like to change the GPU device, add these lines to your model script:

```python
from darts.engines import set_gpu_device
set_gpu_device(N)
```

with `N` being your GPU device number. For example if you have 2 GPUs, you can call `set_gpu_device(0)` or `set_gpu_device(1)`.
