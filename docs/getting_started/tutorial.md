# Tutorial

## Getting started with any model

In order to use DARTS you need to set up the model. It usually includes a few Python scripts called main.py, model.py, and reservoir.py. main.py usually manages the simulation run, the time stepping, and the output. model.py usually contains the description of the physics, and all the parameters required to define it. reservoir.py usually describes the simulation domain, computational grid and the static properties assigned to them. They are usually organized in a way that main.py is a startup scripts where the model is created, then it calls for the model constructor or some other model methods overloaded in model.py which then creates a reservoir and calls its methods defined in reservoir.py.

Apart of these three Python scripts, the model folder may contain extra scripts used for model description or post-processing, or benchmarking purposes. It also may include a mesh folder, or just *.geo, *.msh, *.grdecl files placed in the root folder. They define computational grids used in the model. After the simulation run the program may create a separate folder for the output files.

Plenty of different models are located in [darts-models](https://gitlab.tudelft.nl/darts/darts-models) repository. As an example, we can run [Unstructured_fine](https://gitlab.tudelft.nl/darts/darts-models/-/tree/master/Unstructured_fine) model.

1. Run [main.py](https://gitlab.tudelft.nl/darts/darts-models/-/blob/master/Unstructured_fine/main.py)
2. In the command line you may see the program output that usually consist of two parts: first - pre-processing output, second - the output from time steps. Depending on the size of the computational grid  the pre-processing part may take from a couple of seconds up to an hour. If you see the repeating output from time steps afterward, it means that the program is working fine.
3. In the end of the program you may see a picture plotted in python with some key calculated property or the result of the comparison of the calculated solution against other data. The standard output of the program may be found in a new folder called 'sol_*' which contains VTK snapshots of every times step. However, they are not written by default in every model.

## Configure hardware usage

To set the number of threads (CPU cores) to be used, add:  
<code>
from darts.engines import set_num_threads  <br>
set_num_threads(NT)  <br>
</code>  
All cores used by default.

Turn on GPU usage in calculation by adding the next lines at the start of the python script:
Add palatform='gpu' at set_physics call, for example:
<code>super.set_physics(physics, platform='gpu')

If you would like to change the GPU device, add these lines to your model script:  
<code>
from darts.engines import set_gpu_device  <br>
set_gpu_device(N)  <br>
</code>
with N is your GPU device number. For example if you have 2 GPUs, you can call `set_gpu_device(0)` or `set_gpu_device(1)`.
