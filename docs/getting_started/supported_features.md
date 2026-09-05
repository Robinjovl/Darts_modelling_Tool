# Supported features

There is a list of supported/non-supported features:

* Configurations/parallelization:
  * MPFA and geomechanics are not supported with multithreaded (OpenMP) configuration
  * MPFA and geomechanics are not supported with GPU configuration
  * Adjoint gradients feature is not supported with iterative linear solvers and is not parallelized with either OpenMP or GPU
  * Direct linear solvers are not parallelized (https://gitlab.com/open-darts/open-darts/-/issues/40), so a multithreaded (OpenMP) build speeds up the iterative solvers only
  * GPU platform is supported only for Linux
  * MacOS is not supported
  * ARM platform is not officially supported
* Linear solvers (see [Nonlinear and linear solvers](../technical_reference/solvers.md) for the full picture)
  * The open-source build has two families of *outer* Krylov driver, and they are not interchangeable parts:
    HYPRE's, reached only through `MGRSolverSpec`, and open-darts' own, used by every other spec. HYPRE is
    still used inside the in-tree stack -- BoomerAMG is the pressure stage of CPR and FS-CPR -- so the split
    is at the level of the outer solver, not of every component
  * The two cannot be composed: `GMRESSolverSpec(prec=MGRSolverSpec(...))` is rejected with a `ValueError`
    when the solver is built. MGR already embeds its own FlexGMRES, so assign it directly
  * They cannot all be switched during a run either: `AdaptiveSolverSpec` accepts only engine-resident CPU
    registry specs as candidates. A `GPUSolverSpec` or a Python-resident solver (`PETScSolverSpec`,
    `PardisoSolverSpec`) raises `TypeError` at construction
  * There is no way to borrow a preconditioner across the two families -- neither an in-tree Krylov driven by
    a HYPRE MGR preconditioner, nor an MGR cycle preconditioned by an in-tree component
* Mesh
  * A mesh should contain at least 2 cells (to have at least one cell-cell connection), so 1-cell mesh is not supported
  * The mesh format gmsh 2.1 for unstructured mesh is supported; format 4 is not (due to thirdparty/MeshIO)
* Inclusion of specific potential energy in the energy conservation equation
  * To consider specific potential energy in the energy conservation equation:
      1. Open this Python module `darts>reservoirs>reservoir_base.py`
      2. In the constructor of the class `ReservoirBase`, the variable `grav_acc_for_spe` is defined, which is the gravitational acceleration used for calculation of specific potential energy.
      3. The value is zero by default. Change the value of this variable by overwriting it for your desired planet in m/^s (Earth: 9.80665)
