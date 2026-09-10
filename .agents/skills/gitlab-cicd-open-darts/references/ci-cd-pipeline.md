# Skill: CI/CD Pipeline

## Overview

open-DARTS uses **GitLab CI/CD** with a multi-stage pipeline. The entry point is
`.gitlab-ci.yml`, which includes job definitions from `.cicd/jobs/`.

---

## Pipeline Stages

```
pre_commit → build_images → build → test → deploy → ingest
```

| Stage | Purpose |
|---|---|
| `pre_commit` | Code quality (Ruff, YAML/TOML validation) |
| `build_images` | Docker/Apptainer container images |
| `build` | C++ + Python wheel builds (Linux/Windows/GPU/Valgrind) |
| `test` | Regression tests (Linux/Windows/GPU) + Valgrind checks |
| `deploy` | PyPI, TestPyPI, Zenodo, GitLab Pages, Apptainer registry |
| `ingest` | Code ingestion for MCP server |

---

## CI Job Files (.cicd/jobs/)

| File | Description |
|---|---|
| `pre-commit.yml` | Ruff linting + formatting + hygiene checks |
| `build-images.yml` | Docker/Apptainer image building |
| `build-linux-base.yml` | Base Linux build template |
| `build-linux.yml` | Linux builds (Python 3.10–3.13, with/without ODLS) |
| `build-windows.yml` | Windows builds (Python 3.10–3.13) |
| `build-linux-gpu.yml` | GPU builds for Linux |
| `build-linux-valgrind.yml` | Valgrind-instrumented debug builds |
| `test-linux.yml` | Linux test suite |
| `test-windows.yml` | Windows test suite |
| `test-linux-gpu.yml` | GPU tests |
| `valgrind-check.yml` | Valgrind memory leak detection |
| `upload-wheels.yml` | Wheel upload to SMB/artifacts |
| `deploy.yml` | PyPI, Zenodo, GitLab Pages, Apptainer deploy |
| `gitingest.yml` | Code ingestion for MCP server |

---

## Trigger Rules

Jobs trigger based on these conditions:

| Condition | Triggers |
|---|---|
| Protected branch push | All Python 3.11 jobs |
| Merge request | All Python 3.11 jobs |
| Push to `main` | All Python versions (3.10–3.13) |
| Version tag (`v#.#.#`) | All builds + deploy to PyPI/Zenodo |
| `TEST_ALL_PYTHONS=1` | All Python version builds + tests |
| `TEST_CUSTOM_BRANCH=1` | Python 3.11 build + test |
| `UPLOAD_TEST_PYPI=1` | Upload to TestPyPI |
| `DOCS_PAGES=1` | Build + deploy documentation |
| `RUN_APPTAINER_DEPLOY=1` | Push Apptainer image to registry |

---

## Build Matrix

### Linux Builds

| Job | Python | Solvers | Notes |
|---|---|---|---|
| `build-linux-3.11` | 3.11 | bos_solvers (`-a`) | Default for MRs |
| `build-linux-3.11-ODLS` | 3.11 | openDARTS | Default for MRs |
| `build-linux-3.10`, `3.12`, `3.13` | 3.10, 3.12–3.13 | Both variants | `main` / tags |
| `build-linux-3.11-valgrind` | 3.11 | — | Debug + Valgrind flags |
| `build-linux-gpu` | 3.11 | GPU bos_solvers | CUDA build |

### Windows Builds

Same Python version matrix as Linux, builds with MSVC.

---

## Test Matrix

| Job | What it tests |
|---|---|
| `test-linux-3.11-ODLS` | Model suite + interpolators + discretizer (ODLS) |
| `test-linux-3.11` | Same with bos_solvers |
| `valgrind-check` | Memory leak detection |
| `test-linux-gpu` | GPU-specific tests |

Test sequence in each job:
1. `cd models && darts run_test_suite2.py LOG`
2. `cd tests/engines/src/interpolation && darts test_interpolation.py`
3. `cd discretizer/tests/compare_discretizers && darts main.py`
4. Archive PKL artifacts

---

## Deployment

### PyPI Release

Triggered by version tags (`v1.2.3`):
1. Collects wheels from all build jobs (Linux + Windows, all Python versions)
2. Renames Linux wheels for PyPI compatibility
3. Uploads via `twine`

### TestPyPI

Manual trigger with `UPLOAD_TEST_PYPI=1`.

### GitLab Pages (Documentation)

Triggered on `main` pushes or version tags:
1. Installs `open-darts` with `.[docs]`
2. Runs `sphinx-build -b html . public`
3. Publishes to GitLab Pages

### Zenodo

Triggered on version tags: archives source code and publishes DOI.

### Apptainer Image

Manual trigger with `RUN_APPTAINER_DEPLOY=1`: pushes Apptainer `.sif` image to
GitLab Container Registry.

---

## Running CI Jobs Locally

### Pre-commit (linting)

```bash
pip install pre-commit ruff
FILES="$(git ls-files -- 'darts' 'tests' 'helper_scripts' '.cicd' \
    '.gitlab-ci.yml' '.pre-commit-config.yaml' 'pyproject.toml' \
    | grep -E '\.(py|ya?ml|toml)$')"
pre-commit run --files $FILES --show-diff-on-failure --color always
```

### Build (mirrors CI build job)

```bash
./helper_scripts/build_darts_cmake.sh -c -w -t -m -j 8
```

### Test (mirrors CI test job)

```bash
cd models && darts run_test_suite2.py LOG && cd ..
cd tests/engines/src/interpolation && darts test_interpolation.py && cd ../../../..
cd discretizer/tests/compare_discretizers && darts main.py && cd ../../..
```

### Documentation

```bash
pip install .[docs]
cd docs && sphinx-build -b html . public
```

---

## Artifacts

| Artifact | Location | Retention |
|---|---|---|
| Python wheels | `dist/*.whl` | 1 week |
| Build logs | `*.log` | 1 week |
| Test logs | `models/_logs/*.log` | 1 week |
| Valgrind logs | `models/_valgrind_logs/*.log` | 1 week |
| PKL archives | `models/pkl_lin.tar.gz` | 1 week |
| Linting output | `linting_output.log` | 1 week |
| Documentation | `public/` | Permanent (Pages) |

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `RUN_BOS_JOBS` | Run the proprietary BOS-solvers CI twins (`build`/`test-linux-BOS`, `build`/`test-windows-BOS`). Default `"1"` (declared in `.gitlab-ci.yml`); set to anything else to skip all four. Override per-pipeline from a schedule or manual run. |
| `TEST_ALL_PYTHONS` | Build/test all Python versions |
| `TEST_CUSTOM_BRANCH` | Enable build/test for non-protected branches |
| `TEST_GPU` | Enable GPU tests |
| `UPLOAD_TEST_PYPI` | Upload to TestPyPI |
| `DOCS_PAGES` | Build and deploy documentation |
| `RUN_APPTAINER_DEPLOY` | Deploy Apptainer image |
| `APPEND_VTUNE_MODEL` | Additional models for VTune profiling |
| `SMBNAME`, `SMBLOGIN`, `SMBPASS` | SMB credentials for bos_solvers |
| `PYPIUSER`, `PYPIPWD` | PyPI upload credentials |
