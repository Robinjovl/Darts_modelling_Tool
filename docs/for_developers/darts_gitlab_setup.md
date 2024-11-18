# How is built openDARTS?

The source code of openDARTS is located in a repository hosted in [gitlab.com](https://gitlab.com/open-darts/open-darts) and is built and tested automatically via continuous integration / continuous deployment (CI/CD) service. openDARTS is a python package with binary extensions, built from C++/OpenMP/CUDA code. The python part of openDARTS is located in the folder *darts*. The extension engines lives in the folder *engines*.

The build process reminds a chain. First, external libraries (thirdparty dependencies and linear solvers) are built. They are compiled as static libraries and are not aware of python. Then, binary extensions (engines and discretizer) is built. They are linked to a specific python version. Finally, the python package along with extensions is packed into a python wheel. In order to make sure that the package works as expected, a test suite is executed for every compiled wheel in darts-models. Each of these steps is performed as a Gitlab pipeline. Once changes are made to the repository (push, merge request), its pipeline is triggered. If it successfully finishes, then the pipeline of the next link in the chain is triggered, and so one until the whole package is built and tested. The pipeline status of each of the steps described above can be monitored via badges on the main page.

openDARTS wheels are compiled for Windows and Linux platforms, for Python 3.7 - 3.11. Therefore, starting from engines extensions and for every subsequent step, the pipeline consists of 8 independent jobs. In this way, every version of openDARTS package is verified to work as expected. Intermediate build results along with python wheels can be downloaded as artifacts generated during CI/CD jobs.

In the models folder, one can find examples of simulations that serve as regression tests and provide an excellent starting point for understanding how to work with openDARTS. Each folder includes a model.py file, defining the model, a main.py file orchestrating the simulation, and a set of .pkl files containing reference results.

## More information

Visit the [openDARTS wiki](https://gitlab.com/open-darts/open-darts/-/wikis/home).
