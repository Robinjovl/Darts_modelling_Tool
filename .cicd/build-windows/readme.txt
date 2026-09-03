# ---------------------------------------------------------------------------
# How to compile and run DARTS in the docker container (locally)
# ---------------------------------------------------------------------------


# 1. Start the Docker Desktop app with admin priveleges (should appear in the sys tray).
# 2. Run a command line (cmd) with admin priveleges.
# 3. Run the commands below.
# Note: If you want to leave the container running but detach from it, press Ctrl+P then Ctrl+Q.

# run bare MS container for testing purposes
#docker run -it mcr.microsoft.com/windows/servercore:ltsc2022 cmd

# build a docker image from a Dockerfile
docker build -t vs-buildtools .

# run the docker image, allocate 16G RAM and run command line in the docker
docker run -it -m 16g vs-buildtools cmd

# create and activate a conda environment
conda create -n darts python=3.11 --yes
conda init
# need to restart the shell, with the compiler environment
cmd /k "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" "&" cmd
conda activate darts

# download, compile and install open-DARTS
git clone --branch development --recursive https://gitlab.com/open-darts/open-darts.git
cd open-darts
helper_scripts\build_darts_cmake.bat -w -p -j 10
# helper_scripts\install_darts.bat  # with default gmsh from PyPI
pip install . --ignore-installed gmsh  # custom gmsh compiled from source

# run a test
cd models\cpg_sloping_fault
python main.py
cd ..\..

# run all tests
cd models
python run_test_suite2.py

# ---------------------------------------------------------------------------
# How this image is used by CI/CD
# ---------------------------------------------------------------------------

# The Windows pipeline runs inside this container instead of directly on the
# runner host. Jobs and their runner tags:
#
#   windows-image-builder  (.cicd/jobs/build-images.yml)   tag: windows-shell
#       Builds and pushes $CI_REGISTRY_IMAGE/windows-builder. Kaniko cannot
#       produce Windows images, so this one job still runs on the host with
#       `docker build`. It hashes the files listed in DOCKER_HASH_FILES and
#       skips the build when a tag "sha-<hash>" is already in the registry;
#       set UPDATE_DOCKER_IMAGE=1 to force a rebuild.
#
#   build-windows / build-windows-ODLS  (.cicd/jobs/build-windows.yml)
#   test-windows  / test-windows-ODLS   (.cicd/jobs/test-windows.yml)
#                                                          tag: windows_docker
#       Run in the container. The image to use comes from the
#       WINDOWS_BUILDER_IMAGE variable in .gitlab-ci.yml (defaults to
#       .../windows-builder:latest); override it to pin a sha- tag.
#
# Runner requirements for the "windows_docker" tag:
#   - GitLab Runner with executor = "docker-windows"
#   - Docker daemon in Windows-container mode; host OS build must match the
#     image base (mcr.microsoft.com/windows/servercore:ltsc2022) or the
#     container will not start.
#   - Enough memory per container for the MSVC build, e.g. in config.toml:
#         [runners.docker]
#           image = "mcr.microsoft.com/windows/servercore:ltsc2022"
#           memory = "16g"
#           cpus = "8"
#   - Registry credentials: either `docker login` on the host or
#     DOCKER_AUTH_CONFIG set as a CI variable, so the image can be pulled.

# Note: the GitLab docker-windows executor ignores the image ENTRYPOINT and
# starts the job script in PowerShell directly. Build jobs therefore call
# helper_scripts\build_vars.ps1 from the checked-out repo to import the
# MSVC environment. It is called from the repo rather than baked into the
# image, so editing it does not invalidate the image content hash and
# force a full rebuild.

# Build the CI image locally exactly as the pipeline does:
#   docker build -m 16g -t windows-builder -f .cicd/build-windows/Dockerfile .cicd/build-windows
