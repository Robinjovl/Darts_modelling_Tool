# PETSc solvers

There is an option to use PETSc linear solvers. We use a Python interface provided by petsc4py.
Exposure of a jacobian in open-DARTS to Python makes it possible.
PETSc is available only on Linux, but WSL could be used to make it run on Windows machines.

Please check these models to get how to enable PETSc:
- cpg_sloping_fault (`case_base.py`)
- 1ph_1comp_poroelastic_analytics (`model.py`)

###

### PETSc INSTALLATION

```
conda install -c conda-forge mpi4py
conda install -c conda-forge petsc
conda install -c conda-forge petsc4py
```

### TESTING PETSc

This is optional, but recommended to do before the first usage of PETSc.

```
# get PETSC sources
git clone https://github.com/petsc/petsc.git
cd petsc
git checkout v3.23.0
# and run tests
cd ./src/binding/petsc4py
python test/runtests.py
```
