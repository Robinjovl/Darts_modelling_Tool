 # License information for openDARTS
 
 ## openDARTS (source)
openDARTS source code is distributed under [Apache2.0 license](LICENSE). 

This license **DOES NOT** apply to any of the components listed below [Thirdparty dependencies](#thirdparty-dependencies).

## openDARTS (binaries)
Any openDARTS binaries, including the ones in [pypi](https://pypi.org/project/open-darts/), are distributed under [GPLv3](http://www.gnu.org/licenses/gpl.html). 

## Thirdparty dependencies

- [Pybind11](https://github.com/pybind/pybind11) 
Python binding with C++. Used to expose several components of C++ implementation of openDARTS to python. 
[License](https://github.com/pybind/pybind11/blob/master/LICENSE)

- [SuperLU](https://github.com/xiaoyeli/superlu)
Direct linear solver for systems with sparse matrices. 
Used in the darts-linear-solvers module that implements linear algebra functionality for openDARTS. Note that openDARTS does not use SuperLU_DIST version.
[License](https://github.com/xiaoyeli/superlu/blob/master/License.txt)

- [PorePy](https://github.com/pmgbergen/porepy)
Analytic solution for poroelasticity, used in [darts-models](/open-darts/open-darts/-/tree/development/darts-models/1ph_1comp_poroelastic_analytics). 
[License GPLv3](https://github.com/pmgbergen/porepy/blob/develop/LICENSE)

- [MshIO](https://github.com/qnzhou/MshIO/)
Used for unstructured grid processing in darts-discretizer.
The source code is located [darts-discretizer](/open-darts/open-darts/-/tree/development/darts_discretizer/darts-discretizer/src/mesh/mshio).
It is not used in darts-models yet. 
[License Apache 2.0](https://github.com/qnzhou/MshIO/blob/main/LICENSE)
 
- [Open Porous Media project](https://opm-project.org/)
The code for Corner Point Grid preprocessing used in darts-discretizer.
The source code is located [darts-discretizer](/open-darts/open-darts/-/tree/development/darts_discretizer/darts-discretizer/src/opm).
It is not used in darts-models yet. [License GPLv3](http://www.gnu.org/licenses/gpl.html)

## Data 
The data used as input for the models is distributed under CC0 Creative Commons Public Domain Dedication license.
