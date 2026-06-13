# Skill: Debugging and Profiling open-DARTS

## Overview

open-DARTS provides several mechanisms for debugging and profiling:
- **Debug builds** via CMake (`-D CMAKE_BUILD_TYPE=Debug`)
- **Valgrind** memory checking (custom build flags + CI job)
- **Intel VTune** CPU profiling (hotspots, top-down analysis)
- **Built-in timer system** (`timer_node` in C++ engines)
- **Address Sanitizer** (MSVC, commented-out option)

---

## Conda Environment Policy

Use a prompt-defined conda environment when one is provided. Otherwise, create
and activate one session-level environment at the first build, install, test,
debug, docs, or lint step, then reuse it across all skills for the rest of the
session.

Default session environment:

```bash
conda create -y -n open-darts-session python=3.11
conda activate open-darts-session
```

Keep the debug or Valgrind build, editable install, reproduced model runs, and
any follow-up tests in the same session environment so symbols and Python
bindings stay aligned.

---

## Debug Build

### Linux/macOS

```bash
./helper_scripts/build_darts_cmake.sh -c -d Debug -j 8
```

Or manually:

```bash
mkdir -p build && cd build
cmake -D CMAKE_BUILD_TYPE=Debug -D OPENDARTS_CONFIG=ST ..
make install -j 8
cd ..
pip install --no-deps -e .
```

Debug builds use `-O0` optimization and include debug symbols. Useful for
stepping through C++ code with GDB/LLDB.

### Attaching a Debugger

```bash
# GDB
gdb --args darts <script_path_name>

# CUDA-GDB for GPU builds or CUDA kernels
cuda-gdb --args darts <script_path_name>

# Or attach to running process
gdb -p <pid>
```

For pybind11 modules, break on C++ function names directly.

---

## Valgrind Memory Checking

Valgrind detects memory leaks, uninitialised reads, and invalid accesses in the
C++ extensions.

### Build for Valgrind

```bash
./helper_scripts/build_darts_cmake.sh -c -v -j 8
```

This sets:
- `CMAKE_BUILD_TYPE=Debug`
- `ENABLE_VALGRIND=ON`
- SIMD limited to SSE4.2 (`-march=x86-64-v2 -mno-avx512f`) to avoid Valgrind
  SIGILL on AVX-512 instructions
- `-g -O2` flags (symbols + moderate optimization)

### Run Valgrind Manually

```bash
valgrind --leak-check=full --track-origins=yes \
    darts <script_path_name>
```

### Run Valgrind Check Script (CI-compatible)

```bash
python helper_scripts/valgrind_check.py
```

This runs pre-defined models under Valgrind and writes logs to
`models/_valgrind_logs/`.

### CI Valgrind Pipeline

Defined in `.cicd/jobs/valgrind-check.yml` and
`.cicd/jobs/build-linux-valgrind.yml`:
1. Build with Valgrind support (`-v` flag)
2. Install the wheel
3. Run `helper_scripts/valgrind_check.py`
4. Artifacts: `models/_valgrind_logs/*.log`, `models/_valgrind_logs/*.txt`

---

## Intel VTune Profiling

VTune is used for CPU performance analysis — identifying hotspots, call stacks,
and top-down microarchitecture bottlenecks.

### Prerequisites

- Intel VTune Profiler installed and `vtune` on PATH
- Release build recommended for realistic profiling

### Run VTune Profiling Script

```bash
python helper_scripts/vtune_profiling.py
```

Default model: `SPE11b`. Add more models via environment variable:

```bash
APPEND_VTUNE_MODEL="MyModel1,MyModel2" python helper_scripts/vtune_profiling.py
```

### VTune Output

Logs go to `models/_vtune_logs/`:
- `<model>.log` — stdout from profiled run
- `<model>_summary.csv` — VTune summary report
- `<model>_hotspots.csv` — function-level hotspot report
- `<model>_topdown.csv` — top-down microarchitecture report
- `vtune_mem_<model>/` — raw VTune result directory

### Manual VTune Collection

```bash
vtune -collect hotspots -data-limit=0 \
    -r models/_vtune_logs/vtune_result \
    darts <script_path_name>

# Generate reports
vtune -report summary -format csv -r models/_vtune_logs/vtune_result \
    -report-output summary.csv
vtune -report hotspots -format csv -r models/_vtune_logs/vtune_result \
    -report-output hotspots.csv
vtune -report top-down -format csv -r models/_vtune_logs/vtune_result \
    -report-output topdown.csv
```

---

## Built-in Timer System

The C++ engines include a `timer_node` class that tracks time spent in different
phases of the simulation. After running a model, timer data is available in
Python:

```python
model.run()
model.physics.engine.timer.print()  # Print timing breakdown
```

Key timer nodes:
- Initialization
- Simulation loop
- Newton updates / Jacobian assembly
- Linear solver
- VTK output

---

## Address Sanitizer (MSVC only)

For Windows Debug builds, MSVC Address Sanitizer support is available but
commented out in `CMakeLists.txt`. To enable:

1. Uncomment the `/fsanitize=address` block in `CMakeLists.txt`
2. Copy `clang_rt.asan*.dll` from the Visual Studio installation to the `darts/`
   package directory
3. Rebuild in Debug mode

---

## Python-Level Profiling

For profiling the Python layer:

```bash
# cProfile
python -m cProfile -o profile.out "$(command -v darts)" <script_path_name>
python -c "import pstats; p = pstats.Stats('profile.out'); p.sort_stats('cumulative').print_stats(30)"

# line_profiler (install separately)
kernprof -l -v "$(command -v darts)" <script_path_name>

# py-spy (sampling profiler, low overhead)
py-spy record -o profile.svg -- darts <script_path_name>
```

---

## Compile Commands for IDE Integration

CMake generates `compile_commands.json` in the build directory
(`CMAKE_EXPORT_COMPILE_COMMANDS ON`). Use this for clangd, clang-tidy,
or IDE code navigation:

```bash
# Symlink for IDE auto-discovery
ln -sf build/compile_commands.json .
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Valgrind SIGILL on AVX-512 | Rebuild with `-v` flag (limits to SSE4.2) |
| VTune "not found" | Source VTune env: `source /opt/intel/vtune/latest/vtune-vars.sh` |
| No debug symbols in pybind11 | Use `CMAKE_BUILD_TYPE=Debug` (not RelWithDebInfo) |
| Timer data not available | Ensure `model.run()` completed; check engine type |
| GDB can't find sources | Set `directory` to the repo root in GDB |
