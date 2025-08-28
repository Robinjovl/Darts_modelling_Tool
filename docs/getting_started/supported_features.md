# Supported features

There is a list of supported/non-supported features:

* Configurations/parallelization:
  * MPFA and geomechanics are not supported with multithreaded (OpenMP) configuration
  * MPFA and geomechanics are not supported with GPU configuration
  * Adjoint gradients feature is not supported with iterative linear solvers and is not parallelized with either OpenMP or GPU
  * The ODLS configuration (with direct linear solvers) is not supported with multithreaded (OpenMP) configuration
  * Direct linear solvers are not parallelized (https://gitlab.com/open-darts/open-darts/-/issues/40)
  * GPU platform is supported only for Linux
  * MacOS is not supported
  * ARM platform is not officially supported
* Mesh
  * A mesh should contain at least 2 cells (to have at least one cell-cell connection), so 1-cell mesh is not supported
  * The mesh format gmsh 2.1 for unstructured mesh is supported; format 4 is not (due to thirdparty/MeshIO)