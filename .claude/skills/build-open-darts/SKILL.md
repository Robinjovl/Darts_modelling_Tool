---
name: build-open-darts
description: Build and install open-DARTS C++ extensions and Python wheels across Linux and Windows, including ST/MT/GPU and debug or valgrind-compatible builds. Use when compilation, installation, or packaging is required.
---

# Build open-DARTS

Use this skill for deterministic build and install workflows.

## Steps

1. Select the target mode (local editable install, full wheel build, GPU build, debug build, valgrind build).
2. Run the corresponding helper script from repo root.
3. Verify build outputs (`dist/*.whl`, logs, installed extensions in `darts/`).

## Primary commands

- Linux full build: `./helper_scripts/build_darts_cmake.sh -c -w -p -m -j 8`
- Linux editable install (existing `build/`): `./helper_scripts/install_darts.sh -e`
- GPU build: `./helper_scripts/build_install_darts_gpu.sh -c -p -j 20`

## Reference

Read `references/build.md` for full flags, troubleshooting, and platform details.
