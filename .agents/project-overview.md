# open-DARTS Project Overview

## What is open-DARTS

open-DARTS (Open Delft Advanced Research Terra Simulator) is a Python/C++ hybrid
reservoir simulation framework for energy-transition applications: CO2
sequestration, geothermal energy, reactive transport, and poromechanics.

- **Repository**: <https://gitlab.com/open-darts/open-darts>
- **Documentation**: <https://open-darts.readthedocs.io/en/docs>
- **License**: GPL-3.0-or-later (binaries), Apache-2.0 (source)
- **Python**: >=3.10, package name `open-darts`
- **C++ standard**: C++20
- **Build system**: CMake 3.26+ (C++ extensions) + setuptools (Python wheel)

---

## Repository Layout

```
open-darts/
├── darts/                  # Python package (installed as `open-darts`)
│   ├── models/             # DartsModel, THMCModel, Output
│   ├── physics/            # Physics implementations
│   │   ├── base/           # PhysicsBase, operators_base
│   │   ├── super/          # Compositional (main multi-phase)
│   │   ├── geothermal/     # Geothermal
│   │   ├── chemistry/      # Reactive flow (ElementBased)
│   │   ├── mech/           # Poroelasticity
│   │   ├── deadoil/        # Dead-oil
│   │   ├── blackoil/       # Black-oil
│   │   └── properties/     # Property containers + databases
│   ├── reservoirs/         # Reservoir classes + mesh discretizers
│   ├── engines/            # C++ engine bindings (pybind11)
│   ├── discretizer/        # C++ discretizer bindings
│   ├── input/              # InputData, FluidProps
│   ├── pipes/              # Wellbore modelling
│   └── tools/              # CLI, plotting, HDF5, fracture networks
├── engines/                # C++ engine source code
├── discretizer/            # C++ discretizer source
├── solvers/                # openDARTS linear solvers (SuperLU, Hypre)
├── thirdparty/             # pybind11, SuperLU, Hypre, IPhreeqc (git submodules)
├── models/                 # Example / regression-test models
├── tests/                  # Unit and integration tests
├── docs/                   # Sphinx documentation source
├── helper_scripts/         # Build, install, and utility scripts
├── settings/               # CMake modules
├── .cicd/jobs/             # GitLab CI job definitions
├── CMakeLists.txt          # Top-level CMake configuration
├── pyproject.toml          # Python packaging + Ruff config
├── .pre-commit-config.yaml # Pre-commit hooks
└── .gitlab-ci.yml          # CI/CD entry point
```

---

## Key Python Modules

| Module | Purpose |
|---|---|
| `darts.models.darts_model.DartsModel` | Base simulation model |
| `darts.models.output.Output` | Simulation output (HDF5, VTK) |
| `darts.physics.super.Compositional` | Multi-phase compositional physics |
| `darts.physics.geothermal.Geothermal` | Geothermal physics |
| `darts.physics.chemistry.ElementBasedReactiveFlow` | Reactive flow |
| `darts.physics.mech.Poroelasticity` | Poromechanics |
| `darts.reservoirs.struct_reservoir.StructReservoir` | Structured grid |
| `darts.reservoirs.unstruct_reservoir.UnstructReservoir` | Unstructured grid |
| `darts.reservoirs.cpg_reservoir.CPG_Reservoir` | Corner-point grid |
| `darts.engines.*` | C++ engine bindings |
| `darts.tools.cli` | CLI entry point (`darts` command) |

---

## C++ Extensions (pybind11)

Two compiled extension modules are built from C++ source and installed into the
`darts/` package directory as `.so` (Linux) or `.pyd` (Windows) files:

1. **`darts.engines`** — simulation engines (CPU: ST/MT, GPU), interpolators,
   well models, linear solvers, mesh (`conn_mesh`), timers.
2. **`darts.discretizer`** — discretization methods (TPFA, MPFA).

Build configurations: `ST` (single-thread), `MT` (multi-thread with OpenMP),
`GPU` (CUDA). The `MT` and `GPU` modes require external `bos_solvers`.

---

## CLI

The package provides a `darts` CLI (defined in `darts/tools/cli.py`):

```bash
darts <path-to-script>        # Run a model script
darts run_test_suite2.py LOG  # Run test suite with logging
```

---

## Configuration Files

| File | Purpose |
|---|---|
| `pyproject.toml` | Python metadata, dependencies, Ruff config |
| `CMakeLists.txt` | C++ build configuration |
| `.pre-commit-config.yaml` | Pre-commit hooks (Ruff, hygiene) |
| `.gitlab-ci.yml` | CI/CD pipeline definition |
| `docs/conf.py` | Sphinx documentation config |

---

## Git Conventions

- **Baseline branch**: `development`
- Keep commits focused with brief, descriptive messages.
- Use absolute paths when running commands.
- Run commands non-interactively from the repo root.
