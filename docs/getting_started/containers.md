## Open‑DARTS Apptainer image: pull, build, install, and test

This guide shows how to pull the Open‑DARTS Apptainer builder image from the GitLab Container Registry and then build or install Open‑DARTS inside the image (including running the GPU test suite).

- Repository: `oras://registry.gitlab.com/open-darts/open-darts/apptainer`
- Latest tag: `latest`

CI deploy publishes the image to `oras://$CI_REGISTRY_IMAGE/apptainer:latest` (see `deploy-apptainer-image` in `.cicd/jobs/deploy.yml`). Image build and pruning logic is in `.cicd/jobs/build-images.yml`. GPU build/test flows are mirrored from `.cicd/jobs/build-linux-gpu.yml` and `.cicd/jobs/test-linux-gpu.yml`.

### Prerequisites

- Singularity/Apptainer installed on your host. Below we give description for apptainer.
- For GPU, an NVIDIA host with drivers; use `--nv` when executing inside the image.
- GitLab registry access is private and, therefore, requires Gitlab username:

### Log in (needed for private registries)

```bash
apptainer registry login \
  -u "<GITLAB_USERNAME>" \
  -p "<GITLAB_PASSWORD or GITLAB_PERSONAL_ACCESS_TOKEN>" \
  oras://registry.gitlab.com
```

### Pull the image

Pull latest:

```bash
apptainer pull open-darts-latest.sif \
  oras://registry.gitlab.com/open-darts/open-darts/apptainer:latest
```

Quick checks:

```bash
ls -lah open-darts-*.sif
apptainer exec --nv open-darts-latest.sif source /opt/conda/etc/profile.d/conda.sh
apptainer exec --nv open-darts-latest.sif nvidia-smi || true
# bind repository


```

### Open a shell inside the image

```bash
apptainer shell --nv open-darts-latest.sif
# inside: python, conda, git, nvidia-smi (with --nv), etc.
```

### Build Open‑DARTS from source inside the image

This mirrors the GPU build job that creates a conda env and runs the project helper.

```bash
apptainer exec --nv open-darts-latest.sif bash -lc "
  set -euxo pipefail \ &&
  source /opt/conda/etc/profile.d/conda.sh \ &&
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \ &&
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \ &&
  conda create -n my_env python=3.10 -q -y \ &&
  conda activate my_env \ &&
  ./helper_scripts/build_install_darts_gpu.sh -c -p -j 16
"
```

Notes:
- The helper compiles GPU libs and builds/installs a wheel into the env (adjust `-j` to your cores).
- CI often uses a dedicated env name like `my_env`.

### Install an existing wheel inside the image

If you already have a wheel in `./dist/` (from local build or CI artifacts):

```bash
apptainer exec --nv open-darts-latest.sif bash -lc '
  set -euxo pipefail
  source /opt/conda/etc/profile.d/conda.sh
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  conda create -n tahiti-gpu python=3.10 -q -y
  conda activate tahiti-gpu
  python -m pip install --upgrade pip
  pip uninstall open-darts-flash --yes || true
  pip install ./dist/*.whl --force-reinstall
'
```

### Run the GPU test suite inside the image

This mirrors the test job that activates the env, installs the wheel (if needed), and runs the suite.

```bash
apptainer exec --nv open-darts-latest.sif bash -lc '
  set -euxo pipefail
  source /opt/conda/etc/profile.d/conda.sh
  conda activate tahiti-gpu

  # model tests
  cd models
  darts run_test_suite2.py LOG
  echo "run_test_suite2 returned $?"
  cd ..

  # discretizer tests
  cd discretizer/tests/compare_discretizers
  python main.py
  cd ../../..
  echo "test finished"
'
```

Artifacts (logs/archives) are written under the repository paths, similar to CI runs.

### CI references

- Push to registry: `deploy-apptainer-image` in `.cicd/jobs/deploy.yml` (pushes `oras://$CI_REGISTRY_IMAGE/apptainer:latest`).
- Build image and maintain a `latest` symlink on development branches: `.cicd/jobs/build-images.yml`.
- Build inside image: `build-linux-tahiti-image-3.10-gpu` in `.cicd/jobs/build-linux-gpu.yml`.
- Test inside image: `test-linux-tahiti-image-3.10-gpu` in `.cicd/jobs/test-linux-gpu.yml`.
