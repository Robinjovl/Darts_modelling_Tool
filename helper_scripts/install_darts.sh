#!/bin/bash

# Parse args: -e (editable), --with-deps (install dependencies)
EDITABLE=0
WITH_DEPS=0
JOBS=8

ensure_reaktoro_conda() {
  if python3 - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("reaktoro") else 1)
PY
  then
    return
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "Warning: 'conda' not found; please install Reaktoro manually (conda install -c conda-forge reaktoro)." >&2
    return
  fi

  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Warning: CONDA_PREFIX is empty; activate the target Conda environment before running install_darts.sh to auto-install Reaktoro." >&2
    return
  fi

  local reaktoro_log="$PWD/make_reaktoro.log"
  echo "+ conda install -y -c conda-forge -p ${CONDA_PREFIX} reaktoro" | tee -a "$reaktoro_log"
  conda install -y -c conda-forge -p "${CONDA_PREFIX}" reaktoro 2>&1 | tee -a "$reaktoro_log"
}

for arg in "$@"; do
  case "$arg" in
    -e|--editable)
      EDITABLE=1
      ;;
    -j|--jobs)
      # support -j N and -jN
      if [[ -n "${2:-}" && "$2" =~ ^[0-9]+$ ]]; then
        JOBS=$2
        shift 2
      else
        echo "Error: -j requires a numeric argument"
        exit 1
      fi
      shift
      ;;
    --with-deps)
      WITH_DEPS=1
      ;;
  esac
done

cp CHANGELOG.md darts

# Build C++ extensions first
echo "Building C++ extensions..."
if [ -d "build" ]; then
  cd build
  make -j$JOBS || exit 1
  make install || exit 1
  cd ..
fi

if [ $EDITABLE -eq 1 ]; then
  if [ $WITH_DEPS -eq 1 ]; then
    python3 -m pip install -e .
  else
    python3 -m pip install --no-deps -e .
  fi
else
  # Now build the wheel with extensions included
  python3 -m pip install build
  python3 -m build --wheel
  WHEEL_PATH=$(ls -t dist/*.whl | head -n1)
  if [ $WITH_DEPS -eq 1 ]; then
    python3 -m pip install "file://$PWD/${WHEEL_PATH}"
    # Ensure Conda-level deps (Reaktoro) are present if possible
    ensure_reaktoro_conda
  else
    python3 -m pip install --no-deps --force-reinstall "file://$PWD/${WHEEL_PATH}"
  fi
fi
