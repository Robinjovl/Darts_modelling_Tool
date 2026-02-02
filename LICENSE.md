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
Used in the darts-linear-solvers module that implements linear algebra functionality for openDARTS. Note that open-DARTS does not use SuperLU_DIST version.
[License](https://github.com/xiaoyeli/superlu/blob/master/License.txt)

- [hypre](https://github.com/hypre-space/hypre)
Iterative linear solvers and preconditioners
[License Apache 2.0](https://github.com/hypre-space/hypre/blob/master/LICENSE-APACHE)
[License MIT](https://github.com/hypre-space/hypre/blob/master/LICENSE-MIT)

- [MshIO](https://github.com/qnzhou/MshIO/)
Used for unstructured grid processing in discretizer.
The source code is located [discretizer](discretizer/src/mesh/mshio).
It is not used in models yet.
[License Apache 2.0](https://github.com/qnzhou/MshIO/blob/main/LICENSE)

- [PyGRDECL](https://github.com/BinWang0213/PyGRDECL)
Used in [struct-reservoir](/darts/reservoirs/struct_reservoir.py) to write VTK files.
[License BSD 3-Clause License](https://github.com/BinWang0213/PyGRDECL/blob/master/LICENSE)

- [Fracture-Preprocessing-Code](https://github.com/MakeLikePaperrr/Fracture-Preprocessing-Code).
Used in [darts/tools](/darts/tools/fracture_network).
[License MIT](https://github.com/MakeLikePaperrr/Fracture-Preprocessing-Code/blob/main/LICENSE)

- [iPHREEQC](https://github.com/usgs-coupled/iphreeqc)
Used in models with kinetic reactions.
[Terms of use](https://phreeqcusers.org/index.php/topic,1007.msg2892.html#msg2892)

## Data

The data used as input for the models is distributed under CC0 Creative Commons Public Domain Dedication license.
