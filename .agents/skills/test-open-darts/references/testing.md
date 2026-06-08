# Skill: Testing open-DARTS

## Overview

open-DARTS has a multi-layer testing strategy:
- **Model regression tests** — run simulation models and compare outputs against
  reference PKL (pickle) files.
- **Interpolator tests** — validate the C++ multilinear adaptive interpolators.
- **Discretizer tests** — compare discretization methods.
- **C++ unit tests** — built via CMake with `ENABLE_TESTING=ON`, run with ctest.

---

## Conda Environment Policy

Use a prompt-defined conda environment when one is provided. Otherwise, create
and activate one session-level environment at the first build, install, test,
debug, docs, or lint step, then reuse it across all skills for the rest of the
session.

Default session environment:

```bash
conda create -y -n open-darts-session python=3.10
conda activate open-darts-session
```

Keep the build or editable install and all follow-up test commands in the same
session environment.

---

## Running the Full Test Suite

From the repo root (requires the package to be installed):

```bash
cd models
darts run_test_suite2.py LOG
cd ..
```

The `LOG` argument enables verbose performance checking. Logs are written to
`models/_logs/`.

### Test Categories in run_test_suite2.py

| Category | Models | Notes |
|---|---|---|
| 2-phase compositional | `2ph_comp` | |
| 2-phase dead oil | `2ph_do` | |
| 2-phase geothermal | `2ph_geothermal` | |
| 3-phase compositional | `3ph_comp_w` | |
| 3-phase dead/black oil | `3ph_do`, `3ph_bo` | |
| Brugge | `Uniform_Brugge` | |
| Chemistry | `Chem_benchmark_new` | |
| Geothermal rising | `GeoRising` | |
| Coaxial well | `CoaxWell` | |
| DFM well | DFM tests | CPU only |
| Adjoint | `Adjoint` tests | |
| CPG | `cpg_sloping_fault` | Multiple sub-cases |
| Fracture network | `fracture_network` | Cases 1, 4, 5 |
| Chemistry (Phreeqc) | `chemistry/carbonated_water` | Requires IPhreeqc |
| Poromechanics | `1ph_1comp_poroelastic_*`, `SPE10_mech` | Terzaghi, Mandel, Bai |
| Fault reactivation | `displaced_fault_reactivation` | |

---

## Running Individual Tests

### Single Model

```bash
darts <script_path_name>
```

### Interpolator Tests

```bash
cd tests/engines/src/interpolation
darts test_interpolation.py
```

### Discretizer Tests

```bash
cd discretizer/tests/compare_discretizers
darts main.py
```

### C++ Unit Tests (ctest)

Build with testing enabled, then run:

```bash
cd build
cmake -D ENABLE_TESTING=ON ..
make -j 8
ctest
```

---

## Test Results and Artifacts

- **Logs**: `models/_logs/*.log`
- **Chemistry logs**: `models/chemistry/_logs/chemistry/*.log`
- **Reference data**: PKL files archived via `models/archive_pkl.sh`
- **Performance comparison**: test runner compares simulation results against
  reference PKL files. Regressions are reported as failures.

---

## CI Test Pipeline

Tests run in GitLab CI (`.cicd/jobs/test-linux.yml`) for Python 3.10–3.13 with
two solver configurations:

1. **ODLS** (openDARTS linear solvers) — self-contained
2. **bos_solvers** (private iterative solvers, `-a` flag)

The CI test job sequence:
1. Install the built wheel from `dist/`
2. Run `models/run_test_suite2.py LOG`
3. Run interpolator tests
4. Run discretizer tests
5. Archive PKL artifacts

---

## GPU Tests

GPU tests run separately (`.cicd/jobs/test-linux-gpu.yml`) and require the
`TEST_GPU` environment variable.

---

## Writing New Tests

1. Create a new model directory under `models/` with a `main.py` entry point.
2. The script should be runnable via `darts <script_path_name>` (typically `darts main.py` from the model directory).
3. Add the model name to the test list in `run_test_suite2.py`.
4. Generate reference PKL files on a verified run.
5. The test framework will compare subsequent runs against the reference.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `darts` command not found | Install the package: `pip install -e .` |
| PKL comparison fails | Reference data may be outdated; regenerate with a verified build |
| `gmsh` import error | Install system deps: `apt install libglu1-mesa-dev libxcursor1 libxinerama1` |
| GPU test skipped | Set `TEST_GPU=1` environment variable |
| Chemistry test fails | Ensure IPhreeqc is built (`-p` flag) and Phreeqc databases are present |
