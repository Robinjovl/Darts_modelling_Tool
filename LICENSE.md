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

- [MshIO](https://github.com/qnzhou/MshIO/)
Used for unstructured grid processing in discretizer.
The source code is located [discretizer](discretizer/src/mesh/mshio).
It is not used in darts-models yet.
[License Apache 2.0](https://github.com/qnzhou/MshIO/blob/main/LICENSE)

- [PyGRDECL](https://github.com/BinWang0213/PyGRDECL)
Used in [struct-reservoir](/darts/reservoirs/struct_reservoir.py) to write VTK files.
[License BSD 3-Clause License](https://github.com/BinWang0213/PyGRDECL/blob/master/LICENSE)

- [STREAM](https://www.cs.virginia.edu/stream/ref.html)
Used in [engines](/engines/src/stream.cpp).
[License](https://www.cs.virginia.edu/stream/FTP/Code/LICENSE.txt)

- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
Used in [SVD for poromechanics](/engines/src/mech/matrix.hpp).
[License](https://gitlab.com/libeigen/eigen/-/blob/master/COPYING.APACHE)

## Data

The data used as input for the models is distributed under CC0 Creative Commons Public Domain Dedication license.
