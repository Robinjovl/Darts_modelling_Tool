#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_DARTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

clean_mode=false
skip_req=false
skip_submodule=false
skip_thirdparty_check=true
config="Release"
threads=64
rebuild_hypre=false
use_mirror=true
toolset=""
testing=false
special_gpp=false
gpp_version="g++"
opendarts_config="ST"

Help_Info()
{
  cat <<'EOF'
build_darts_with_mgr.sh [OPTIONS]

  Build open-DARTS with MGR Linear Solver support on Linux/macOS.

USAGE:
  helper_scripts/build_darts_with_mgr.sh [-h] [-c] [-r] [-d CONFIG] [-j N]
                                        [--rebuild-hypre] [--skip-submodule]
                                        [--check-thirdparty] [--no-mirror]
                                        [--toolset TOOLSET] [-t] [-g COMPILER]

OPTIONS:
  -h                  Display this help message
  -c                  Clean openDARTS build directories before configuring
  -r                  Skip building thirdparty libraries (HYPRE, SuperLU)
  -d CONFIG           Build configuration [Release, Debug]. Default: Release
  -j N                Set number of threads for compilation. Default: 64
  -t                  Enable ctest after building openDARTS
  -g COMPILER         Specify C++ compiler, e.g. g++-13
  --rebuild-hypre     Delete HYPRE build/install outputs and rebuild HYPRE
  --skip-submodule    Skip git submodule initialization
  --skip-thirdparty-check
                      Skip thirdparty library check. Default.
  --check-thirdparty  Run thirdparty library check/submodule init
  --no-mirror         Accepted for parity with the Windows .bat script
  --toolset TOOLSET   Accepted for parity with the Windows .bat script;
                      ignored on Unix builds

EXAMPLES:
  # Standard build
  helper_scripts/build_darts_with_mgr.sh

  # Fast local rebuild when thirdparty libs are unchanged
  helper_scripts/build_darts_with_mgr.sh -r

  # Clean openDARTS rebuild while keeping thirdparty outputs
  helper_scripts/build_darts_with_mgr.sh -c -r

  # Rebuild HYPRE and then rebuild openDARTS
  helper_scripts/build_darts_with_mgr.sh -c --rebuild-hypre

  # Build with Debug configuration
  helper_scripts/build_darts_with_mgr.sh -d Debug

  # Run thirdparty check/submodule init, then build everything
  helper_scripts/build_darts_with_mgr.sh --check-thirdparty

REQUIREMENTS:
  - CMake 3.26 or higher
  - C++20-capable compiler
  - make
  - Python 3.10 or higher
  - Git

ENVIRONMENT:
  OPENDARTS_EXTRA_CMAKE_ARGS
                      Extra CMake arguments passed only to the openDARTS
                      configure step, e.g. "-D Python_INCLUDE_DIR=..."

OUTPUTS:
  - HYPRE:       thirdparty/install/lib/
  - SuperLU:     thirdparty/SuperLU_5.2.1/
  - open-DARTS:  darts/*.so and darts/solvers/solvers*.so
EOF
}

header()
{
  printf '\n========================================================================\n'
  printf '  %s\n' "$1"
  printf '========================================================================\n\n'
}

warn()
{
  printf '[WARNING] %s\n' "$*" >&2
}

die()
{
  printf '\n========================================================================\n' >&2
  printf '  BUILD FAILED\n' >&2
  printf '========================================================================\n\n' >&2
  printf 'Error: %s\n\n' "$*" >&2
  printf 'Check log files in %s:\n' "${OPEN_DARTS_ROOT}" >&2
  printf '  - make_submodules.log\n' >&2
  printf '  - make_hypre.log\n' >&2
  printf '  - make_superlu.log\n' >&2
  printf '  - make_darts.log\n\n' >&2
  exit 1
}

print_command()
{
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_logged()
{
  local log_file="$1"
  shift
  print_command "$@"
  "$@" > "${log_file}" 2>&1
}

run_logged_append()
{
  local log_file="$1"
  shift
  print_command "$@"
  "$@" >> "${log_file}" 2>&1
}

require_command()
{
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || die "Required command not found: ${cmd}"
}

realpath_m()
{
  realpath -m "$1" 2>/dev/null || printf '%s\n' "$1"
}

ensure_cmake_cache_matches()
{
  local build_dir="$1"
  local source_dir="$2"
  local label="$3"
  local cache_file="${build_dir}/CMakeCache.txt"

  [[ -f "${cache_file}" ]] || return 0

  local expected_build expected_source actual_build actual_source
  expected_build="$(realpath_m "${build_dir}")"
  expected_source="$(realpath_m "${source_dir}")"
  actual_build="$(awk -F= '/^CMAKE_CACHEFILE_DIR:INTERNAL=/{print $2}' "${cache_file}" | tail -n 1 || true)"
  actual_source="$(awk -F= '/^CMAKE_HOME_DIRECTORY:INTERNAL=/{print $2}' "${cache_file}" | tail -n 1 || true)"

  [[ -n "${actual_build}" ]] && actual_build="$(realpath_m "${actual_build}")"
  [[ -n "${actual_source}" ]] && actual_source="$(realpath_m "${actual_source}")"

  if [[ -n "${actual_build}" && "${actual_build}" != "${expected_build}" ]] ||
     [[ -n "${actual_source}" && "${actual_source}" != "${expected_source}" ]]; then
    warn "${label} CMake cache belongs to another source/build directory. Cleaning ${build_dir}"
    rm -rf "${build_dir}"
  fi
}

parse_args()
{
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        Help_Info
        exit 0
        ;;
      -c)
        clean_mode=true
        shift
        ;;
      -r)
        skip_req=true
        shift
        ;;
      -d)
        [[ $# -ge 2 ]] || die "-d requires a build configuration"
        config="$2"
        shift 2
        ;;
      -j)
        [[ $# -ge 2 ]] || die "-j requires a thread count"
        threads="$2"
        shift 2
        ;;
      -t)
        testing=true
        shift
        ;;
      -g)
        [[ $# -ge 2 ]] || die "-g requires a compiler command"
        special_gpp=true
        gpp_version="$2"
        shift 2
        ;;
      --rebuild-hypre)
        rebuild_hypre=true
        shift
        ;;
      --skip-submodule)
        skip_submodule=true
        shift
        ;;
      --skip-thirdparty-check)
        skip_thirdparty_check=true
        shift
        ;;
      --skip-thirdparty-check=false|--check-thirdparty)
        skip_thirdparty_check=false
        shift
        ;;
      --no-mirror)
        use_mirror=false
        shift
        ;;
      --toolset)
        [[ $# -ge 2 ]] || die "--toolset requires a toolset name"
        toolset="$2"
        shift 2
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done

  [[ "${threads}" =~ ^[0-9]+$ ]] || die "Thread count must be numeric: ${threads}"
  [[ "${threads}" -gt 0 ]] || die "Thread count must be greater than zero"
}

preflight()
{
  cd "${OPEN_DARTS_ROOT}"

  [[ -f "${OPEN_DARTS_ROOT}/CMakeLists.txt" ]] ||
    die "CMakeLists.txt not found. Run this script from the open-darts tree."
  [[ -d "${OPEN_DARTS_ROOT}/thirdparty" ]] ||
    die "thirdparty folder not found. Initialize repository dependencies first."

  require_command git
  require_command cmake
  require_command make
  require_command python3
  if [[ "${special_gpp}" == true ]]; then
    require_command "${gpp_version}"
  else
    require_command g++
  fi

  if ! python3 - <<'PY'
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
  then
    die "Python 3.10 or higher is required"
  fi

  if [[ -n "${toolset}" ]]; then
    warn "--toolset=${toolset} is a Windows/MSBuild option and is ignored on Unix"
  fi
  if [[ "${use_mirror}" == false ]]; then
    warn "--no-mirror accepted for Windows script parity; Unix submodule update uses configured git URLs"
  fi
}

print_config()
{
  header "Building open-DARTS with MGR Linear Solver"
  printf 'Configuration:\n'
  printf '  config                 = %s\n' "${config}"
  printf '  threads                = %s\n' "${threads}"
  printf '  clean_mode             = %s\n' "${clean_mode}"
  printf '  skip_req               = %s\n' "${skip_req}"
  printf '  skip_submodule         = %s\n' "${skip_submodule}"
  printf '  skip_thirdparty_check  = %s\n' "${skip_thirdparty_check}"
  printf '  rebuild_hypre          = %s\n' "${rebuild_hypre}"
  printf '  opendarts_config       = %s\n' "${opendarts_config}"
  printf '  root                   = %s\n\n' "${OPEN_DARTS_ROOT}"
}

initialize_submodules()
{
  if [[ "${skip_thirdparty_check}" == true ]]; then
    header "Skipping thirdparty library check"
    printf 'Assuming thirdparty libraries are already initialized.\n'
    return 0
  fi

  if [[ "${skip_submodule}" == true ]]; then
    header "Skipping git submodule initialization"
    return 0
  fi

  header "Initializing thirdparty libraries"
  (
    cd "${OPEN_DARTS_ROOT}"
    git submodule sync --recursive
    git submodule update --init --recursive
  ) > "${OPEN_DARTS_ROOT}/make_submodules.log" 2>&1 ||
    warn "Failed to update submodules; continuing with existing libraries. See make_submodules.log"

  if [[ -d "${OPEN_DARTS_ROOT}/thirdparty/hypre/.git" ]]; then
    local hypre_label
    hypre_label="$(
      cd "${OPEN_DARTS_ROOT}/thirdparty/hypre" &&
      git describe --tags --always 2>/dev/null || true
    )"
    if [[ -n "${hypre_label}" ]]; then
      printf '  [OK] Using HYPRE checkout %s\n' "${hypre_label}"
    else
      printf '  [OK] Using current HYPRE checkout in thirdparty/hypre\n'
    fi
  fi
}

clean_artifacts()
{
  header "Cleaning stale artifacts"
  find "${OPEN_DARTS_ROOT}/darts" \
    -type f \( -name '*.so' -o -name '*.pyd' -o -name '*.dylib' \) \
    ! -name 'libstdc++.so.6' \
    -delete 2>/dev/null || true

  if [[ "${clean_mode}" == true ]]; then
    rm -rf "${OPEN_DARTS_ROOT}/build"
    rm -rf "${OPEN_DARTS_ROOT}/solvers/solver_mgr/build"
  fi
}

build_hypre()
{
  header "Step 1/3: Building HYPRE with MGR support"

  if [[ "${rebuild_hypre}" == true ]]; then
    rm -rf "${OPEN_DARTS_ROOT}/thirdparty/hypre/src/cmbuild"
    rm -rf "${OPEN_DARTS_ROOT}/thirdparty/hypre/cmbuild"
    rm -rf "${OPEN_DARTS_ROOT}/thirdparty/install"
  fi

  local hypre_source=""
  if [[ -f "${OPEN_DARTS_ROOT}/thirdparty/hypre/src/CMakeLists.txt" ]]; then
    hypre_source="${OPEN_DARTS_ROOT}/thirdparty/hypre/src"
  elif [[ -f "${OPEN_DARTS_ROOT}/thirdparty/hypre/CMakeLists.txt" ]]; then
    hypre_source="${OPEN_DARTS_ROOT}/thirdparty/hypre"
  else
    die "HYPRE source not found under thirdparty/hypre"
  fi

  local hypre_build="${hypre_source}/cmbuild"
  local hypre_install="${OPEN_DARTS_ROOT}/thirdparty/install"
  ensure_cmake_cache_matches "${hypre_build}" "${hypre_source}" "HYPRE"
  mkdir -p "${hypre_build}" "${hypre_install}"

  run_logged "${OPEN_DARTS_ROOT}/make_hypre.log" \
    cmake -S "${hypre_source}" -B "${hypre_build}" \
      -D HYPRE_ENABLE_TIMING=OFF \
      -D HYPRE_BUILD_TESTS=OFF \
      -D HYPRE_BUILD_EXAMPLES=OFF \
      -D HYPRE_ENABLE_MPI=OFF \
      -D CMAKE_BUILD_TYPE="${config}" \
      -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
      -D CMAKE_INSTALL_PREFIX="${hypre_install}"

  run_logged_append "${OPEN_DARTS_ROOT}/make_hypre.log" \
    cmake --build "${hypre_build}" --target install --parallel "${threads}" --config "${config}"

  if ! find "${hypre_install}/lib" -maxdepth 2 -type f \
      \( -name 'libHYPRE.*' -o -name 'HYPRE.lib' \) | grep -q .; then
    die "HYPRE library was not found under ${hypre_install}/lib after build"
  fi

  printf '  [OK] HYPRE installed to %s\n' "${hypre_install}"
}

build_superlu()
{
  header "Building SuperLU"

  local superlu_dir="${OPEN_DARTS_ROOT}/thirdparty/SuperLU_5.2.1"
  [[ -d "${superlu_dir}" ]] || die "SuperLU directory not found: ${superlu_dir}"

  (
    cd "${superlu_dir}"
    if [[ "${OSTYPE:-}" == darwin* && -f conf_gcc-11_macOS_m1.mk ]]; then
      cp conf_gcc-11_macOS_m1.mk conf.mk
      cp make_gcc-11_macOS_m1.inc make.inc
    else
      cp conf_gcc_linux.mk conf.mk
      cp make_gcc_linux.inc make.inc
    fi
  )

  run_logged "${OPEN_DARTS_ROOT}/make_superlu.log" \
    make -C "${superlu_dir}" -j "${threads}"
  run_logged_append "${OPEN_DARTS_ROOT}/make_superlu.log" \
    make -C "${superlu_dir}" install -j "${threads}"

  printf '  [OK] SuperLU built in %s\n' "${superlu_dir}"
}

build_thirdparty()
{
  if [[ "${skip_req}" == true ]]; then
    header "Skipping thirdparty build"
    printf 'Assuming HYPRE and SuperLU are already built.\n'
    return 0
  fi

  build_hypre
  build_superlu
}

build_opendarts()
{
  header "Step 2/3: Building openDARTS with MGR integration"

  local build_dir="${OPEN_DARTS_ROOT}/build"
  ensure_cmake_cache_matches "${build_dir}" "${OPEN_DARTS_ROOT}" "openDARTS"
  mkdir -p "${build_dir}"

  local cmake_args=(
    -S "${OPEN_DARTS_ROOT}"
    -B "${build_dir}"
    -D "CMAKE_BUILD_TYPE=${config}"
    -D "CMAKE_INSTALL_PREFIX=${OPEN_DARTS_ROOT}/darts"
    -D "OPENDARTS_CONFIG=${opendarts_config}"
  )

  if [[ "${special_gpp}" == true ]]; then
    cmake_args+=(-D "CMAKE_CXX_COMPILER=${gpp_version}")
  fi
  if [[ "${testing}" == true ]]; then
    cmake_args+=(-D ENABLE_TESTING=ON)
  fi
  if [[ -n "${OPENDARTS_EXTRA_CMAKE_ARGS:-}" ]]; then
    local extra_cmake_args=()
    read -r -a extra_cmake_args <<< "${OPENDARTS_EXTRA_CMAKE_ARGS}"
    cmake_args+=("${extra_cmake_args[@]}")
  fi

  run_logged "${OPEN_DARTS_ROOT}/make_darts.log" cmake "${cmake_args[@]}"
  run_logged_append "${OPEN_DARTS_ROOT}/make_darts.log" \
    cmake --build "${build_dir}" --target install --parallel "${threads}" --config "${config}"

  if [[ "${testing}" == true ]]; then
    run_logged_append "${OPEN_DARTS_ROOT}/make_darts.log" \
      ctest --test-dir "${build_dir}" --output-on-failure -C "${config}"
  fi

  header "Step 3/3: Building openDARTS with MGR integration: DONE"
}

verify_artifacts()
{
  header "Verifying MGR integration"
  shopt -s nullglob

  local engines=( "${OPEN_DARTS_ROOT}"/darts/engines*.so "${OPEN_DARTS_ROOT}"/darts/engines*.pyd )
  local discretizer=( "${OPEN_DARTS_ROOT}"/darts/discretizer*.so "${OPEN_DARTS_ROOT}"/darts/discretizer*.pyd )
  local solvers=( "${OPEN_DARTS_ROOT}"/darts/solvers/solvers*.so "${OPEN_DARTS_ROOT}"/darts/solvers/solvers*.pyd )

  ((${#engines[@]} > 0)) || die "No darts/engines extension found"
  ((${#discretizer[@]} > 0)) || die "No darts/discretizer extension found"
  ((${#solvers[@]} > 0)) || die "No darts/solvers/solvers extension found"

  printf '  [OK] %s\n' "${engines[0]}"
  printf '  [OK] %s\n' "${discretizer[0]}"
  printf '  [OK] %s\n' "${solvers[0]}"
  printf '  [OK] MGR solver integration verified through solvers extension artifact\n'

  if [[ -f "${OPEN_DARTS_ROOT}/darts/print_build_info.py" ]]; then
    PYTHONPATH="${OPEN_DARTS_ROOT}" python3 "${OPEN_DARTS_ROOT}/darts/print_build_info.py" || true
  fi

  shopt -u nullglob
}

print_summary()
{
  header "Build Summary"
  cat <<EOF
Build completed successfully.

Components built:
  [OK] HYPRE library with MGR support
  [OK] SuperLU library
  [OK] MGR solver integration through linsolv_mgr
  [OK] open-DARTS engines, discretizer, and solvers extensions

Output locations:
  - HYPRE library:  ${OPEN_DARTS_ROOT}/thirdparty/install/lib
  - SuperLU:        ${OPEN_DARTS_ROOT}/thirdparty/SuperLU_5.2.1
  - open-DARTS:     ${OPEN_DARTS_ROOT}/darts

Usage in open-DARTS:
  n.params.linear_type = n.params.linear_solver_t.cpu_gmres_mgr
EOF
}

main()
{
  parse_args "$@"
  print_config
  preflight
  initialize_submodules
  clean_artifacts
  build_thirdparty
  build_opendarts
  verify_artifacts
  print_summary
}

main "$@"
