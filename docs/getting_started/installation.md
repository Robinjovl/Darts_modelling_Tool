# Installation of open-darts

### Using pip

We recommend to use a python environment. Then installing open-darts is as simple as:

```bash
pip install open-darts
```

To build open-darts from source go to [build instructions](https://gitlab.com/open-darts/open-darts/-/wikis/Build-instructions)

### Enabling PHREEQC/Reaktoro support

Chemical models that rely on PHREEQC or the Reaktoro thermodynamic engine require additional third-party components. Use the `-p` flag on the helper scripts to enable their installation. The steps below follow the [official Reaktoro Conda instructions](https://reaktoro.org/installation/installation-using-conda.html):

1. Install Conda/Miniconda if it is not already available on your system.
2. Create (optional) and activate the Python environment that you use for `open-darts`, for example:

   ```bash
   conda create --name darts-chem python=3.10
   conda activate darts-chem
   ```

3. Install Reaktoro into the active environment:

   ```bash
   conda install -c conda-forge reaktoro
   ```

4. From the repository root run the helper script with `-p` to build iPHREEQC and register the Reaktoro dependency (Windows users can call the `.bat` variant):

   ```bash
   ./helper_scripts/build_darts_cmake.sh -p
   ```

   The same flag can be passed through other scripts such as `helper_scripts/build_install_darts_gpu.sh`.

The helper script verifies that `conda` is available and that `CONDA_PREFIX` points to the intended environment before installing Reaktoro. Activate the correct Conda environment prior to invoking the script; a detailed installation log is written to `make_reaktoro.log`.
