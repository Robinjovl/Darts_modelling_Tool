cmake_minimum_required (VERSION 3.26)
project (openDARTS LANGUAGES CXX VERSION 0.2)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure runtime path for finding dynamic libraries
set(CMAKE_INSTALL_RPATH $ORIGIN)  # shared libraries will be searched in darts folder
set(CMAKE_MACOSX_RPATH 1)

# Set DARTS installation folder
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/darts/")

# Disable C++ 11 ABI to support old libraries
set(SET_CXX11_ABI_0 TRUE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # useful for code tools like clangd or IDEs

# Turn off valgrind support by default
option(ENABLE_VALGRIND "Enable Valgrind instrumentation (Debug build with symbols)" OFF)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    if(ENABLE_VALGRIND)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Default build type" FORCE)
    else()
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Default build type" FORCE)
    endif()
endif()

if(ENABLE_VALGRIND)
    message(STATUS ">>> Valgrind build: adding -g -O2 with Valgrind-compatible SIMD flags")
    # Use -g -O2 for optimization with debug symbols
    # Limit SIMD to SSE4.2 (supported by Valgrind 3.24) to avoid SIGILL on AVX-512
    # -mno-avx512f disables AVX-512 foundation instructions
    # -march=x86-64-v2 targets SSE4.2 level (compatible with Valgrind)
    set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -g -O2 -march=x86-64-v2 -mno-avx512f")
    set(CMAKE_C_FLAGS_DEBUG            "${CMAKE_C_FLAGS_DEBUG}   -g -O2 -march=x86-64-v2 -mno-avx512f")
endif()

# Load cmake modules -----------------------------------------------------------
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/settings/cmake/modules/")

include(FeatureSummary)
include(GetGitRevisionDescription) # to extract git information (available in source)
include(BundleCMakeStaticLibs) # to bundle a static lib and its dependencies into a single static lib
# ------------------------------------------------------------------------------


# Get environment information for the build and IDEs ---------------------------

# Give the user some feedback on what is going on
message(CHECK_START "Defining build settings")  # build settings

# Get cmake source dir
set(opendarts_SOURCE_DIR ${CMAKE_SOURCE_DIR})

# Get hostname and username
cmake_host_system_information(RESULT opendarts_BUILD_HOSTNAME QUERY HOSTNAME)
if(WIN32)
  set(opendarts_BUILD_USERNAME $ENV{USERNAME})
else()
  set(opendarts_BUILD_USERNAME $ENV{USER})
endif()

# Show build information
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  username: ${opendarts_BUILD_USERNAME}")
message(STATUS "  hostname: ${opendarts_BUILD_HOSTNAME}")

# Get git information
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
git_local_changes(GIT_DIRTY)

# Show git information
message(STATUS "  git refspec: ${GITREFSPEC}")
message(STATUS "  git hash: ${GIT_SHA1}")
message(STATUS "  git dirtiness: ${GIT_DIRTY}")

# Check the build type:
#   ST: single threaded (Default)
#   MT: multi threaded
#   GPU: GPU support
if (NOT DEFINED OPENDARTS_CONFIG)
  set(OPENDARTS_CONFIG "ST")
endif()

if (${OPENDARTS_CONFIG} STREQUAL "GPU")
  set(GPU_CONFIG 1)
endif()

if(${OPENDARTS_CONFIG} STREQUAL "ST")
  set(OPENDARTS_CONFIG "ST")
elseif(${OPENDARTS_CONFIG} STREQUAL "MT" OR GPU_CONFIG)
  if(NOT DEFINED BOS_SOLVERS_DIR)
    message(FATAL_ERROR "openDARTS ${OPENDARTS_CONFIG} requires bos_solvers, please use -D BOS_SOLVERS_DIR = <path_to_bos_solvers> to specify the location of the library.")
  endif()
  if (GPU_CONFIG)
    set(CUDA 1)
  endif()
else()
  message(FATAL_ERROR "openDARTS ${OPENDARTS_CONFIG} is an invalid configuration use: ST, MT or GPU.")
endif()

message(STATUS "  Building openDARTS ${OPENDARTS_CONFIG}")

# Determine the type of compiler we have and set compiler flags and other compiler
# specific flags
if(DEFINED BOS_SOLVERS_DIR)
  add_compile_definitions(WITH_HYPRE)
endif()

# Set compiler specific variables
# Used for compiler flags and other cross compiler variables
message(STATUS "  Setting compiler and linker flags for: ${CMAKE_CXX_COMPILER_ID}")
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # Compiler flags
  #add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)
  #set(opendarts_cxx_compiler_flags, "-W3")

  message(STATUS "  Setting MSVC compiler and linker flags")
  # Compiler flags
  set(opendarts_cxx_compiler_flags "/MP;/std:c++20")

  # for VS Sanitizer usage:
  # 1. copy files clang_rt.asan*.dll from the VS folder, for example: C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64
  # to the darts folder, for example: C:\anaconda3\envs\darts_debug\Lib\site-packages\darts
  # 2. uncomment lines below:
  #if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  #  list(APPEND opendarts_cxx_compiler_flags "/fsanitize=address")
  #endif()

  # Linker flags
  set(opendarts_cxx_linker_flags "")

  if(${OPENDARTS_CONFIG} STREQUAL "MT")
    list(APPEND opendarts_cxx_compiler_flags "/openmp")
  endif()

  message("COMPILER FLAGS:" opendarts_cxx_compiler_flags)

  # Compiler specific helper variables

  # Flags cmake to import external libraries as msvc like files: .lib and .dll instead of .a and .so
  set(import_externals_as_msvc TRUE)

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "  Setting GNU compiler and linker flags")
  # Compiler flags
  # We can also switch -fno-lto to -flto to compile everything with this flag.
  # Using LTO as it seems to reduce compile time
  # No need for manual optimization flag, CMake build types automatically handle it.
  # Debug: O0    Release: 03
  set(opendarts_cxx_compiler_flags "-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused;-fPIC;-fno-lto;-pipe;-std=c++20")

  # Linker flags
  set(opendarts_cxx_linker_flags "")

  if(${OPENDARTS_CONFIG} STREQUAL "MT")
    list(APPEND opendarts_cxx_compiler_flags "-fopenmp")
    list(APPEND opendarts_cxx_linker_flags "-fopenmp")
  endif()

  if(SET_CXX11_ABI_0)
    list(APPEND opendarts_cxx_compiler_flags "-D_GLIBCXX_USE_CXX11_ABI=0")
  endif()

  # Compiler specific helper variables (currently not needed)

elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(STATUS "  Setting Clang compiler and linker flags")
  # Compiler flags
  set(opendarts_cxx_compiler_flags "-O3;-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused;-fPIC")

  # Linker flags
  # Reasons:
  #   -Wl,-no_compact: used because in Clang compilers we get linker warning of
  #                    being unable to create compact unwinding. This is relevant
  #                    for efficient exceptions, but this comes from SuperLU, so
  #                    we deactivate compact unwinding warnings.
  set(opendarts_cxx_linker_flags "-Wl,-no_compact_unwind")

  if(${OPENDARTS_CONFIG} STREQUAL "MT")
    list(APPEND opendarts_cxx_compiler_flags "-fopenmp")
    list(APPEND opendarts_cxx_linker_flags "-fopenmp")
  endif()

  if(SET_CXX11_ABI_0)
    list(APPEND opendarts_cxx_compiler_flags "-D_GLIBCXX_USE_CXX11_ABI=0")
  endif()

  # Compiler specific helper variables (currently not needed)

endif()

set(opendarts_cxx_warning_suppression_flags "")
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(opendarts_cxx_warning_suppression_flags
    # --- Reasonable suppressions (keep permanently) ---
    -Wno-attributes         # cross-platform attribute compatibility (pybind11, MSVC attributes)
    -Wno-maybe-uninitialized # GCC false positives in optimized builds
    -Wno-comment            # harmless nested comment warnings
    -Wno-cast-function-type # common with pybind11 callback casts
    # --- Tech debt: to be fixed incrementally (202+154+140+656+194 unique locations) ---
    -Wno-unused-variable       # ~202 unique locations, mostly in engine template instantiations
    -Wno-unused-but-set-variable # ~100 unique locations in engine templates
    -Wno-unused-parameter      # ~50 unique locations, mostly in virtual method overrides
    -Wno-shadow                # ~154 unique locations, needs per-location review
    -Wno-sign-compare          # ~140 unique locations, int vs size_t in loops
  )
endif()

list(APPEND opendarts_cxx_compiler_flags ${opendarts_cxx_warning_suppression_flags})

message(STATUS "   Compiler flags: ${opendarts_cxx_compiler_flags}")
message(STATUS "   Linker flags  : ${opendarts_cxx_linker_flags}")
message(STATUS "   Warning suppressions: ${opendarts_cxx_warning_suppression_flags}")

# Setup options
option(ENABLE_TESTING "Build and copy testing stuff" OFF)  # by default tests are not built, to save time
add_feature_info(ENABLE_TESTING ENABLE_TESTING "Build and copy testing stuff")  # add option as feature for pretty display
option(ONLY_SOLVERS "Build only solvers" OFF) # by default all project is built
add_feature_info(ONLY_SOLVERS ONLY_SOLVERS "Build only solvers")

# Interpolator compile profile -------------------------------------------------
# Controls which (i_t, algorithm) interpolator variants are stamped per
# (N_DIMS, N_OPS) pair. Reducing the variant count is the most effective lever
# against the cc1plus OOM hit on heavy py_operator_set_interpolator_*_part*.cpp
# translation units (~1.5–4 GB resident per TU at -O2/-O3 today).
#   MINIMAL: multilinear_adaptive uint64 only. Drops linear; any model that
#            explicitly sets itor_type='linear' must switch to 'multilinear'.
#   FULL:    multilinear_adaptive uint64 + linear_adaptive uint64. No behaviour
#            change vs. legacy other than removing the (compile-only) uint32
#            multilinear specialisation; Python first-try at uint32 falls
#            through to uint64 via the existing try/except (one-time cost at
#            interpolator construction).
set(OPENDARTS_INTERPOLATOR_PROFILE "FULL" CACHE STRING
    "Interpolator compile profile: MINIMAL (multilinear uint64 only) or FULL (+linear uint64).")
set_property(CACHE OPENDARTS_INTERPOLATOR_PROFILE PROPERTY STRINGS MINIMAL FULL)
if(NOT OPENDARTS_INTERPOLATOR_PROFILE MATCHES "^(MINIMAL|FULL)$")
    message(FATAL_ERROR "OPENDARTS_INTERPOLATOR_PROFILE must be MINIMAL or FULL (got '${OPENDARTS_INTERPOLATOR_PROFILE}').")
endif()
message(STATUS "  Interpolator profile: ${OPENDARTS_INTERPOLATOR_PROFILE}")

# Maximum number of OBL parameter-space dimensions for interpolator template
# instantiation. The recursive_exposer loops produce one class per (N_DIMS,
# N_OPS) pair for N_DIMS in 1..OPENDARTS_MAX_DIMS, so this is also the dominant
# knob on compile-time memory: interpolate_with_derivatives unrolls O(2^N_DIMS)
# per instantiation. Default 8 matches the historical hardcoded value.
#
# This single variable drives BOTH:
#   - MAX_DIMS in interpolation_config.h (per-TU define on the interpolators target)
#   - MAX_NC   in engines/src/globals.h  (global define; bounds the engine_nc_*
#                                         and engine_super_* component count loop)
# The two were always set to the same value in practice; mapping them to one
# cmake variable removes the silent NC=8 thermal mismatch (MAX_DIMS must be at
# least MAX_NC + 1 for thermal models — keep them equal here, document the
# constraint, and increase OPENDARTS_MAX_DIMS if a thermal-at-max-NC model is
# needed).
set(OPENDARTS_MAX_DIMS "8" CACHE STRING
    "Maximum number of OBL parameter-space dimensions and engine components (typical: 6–8). Drives both MAX_DIMS (interpolators) and MAX_NC (engines).")
if(NOT OPENDARTS_MAX_DIMS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "OPENDARTS_MAX_DIMS must be a positive integer (got '${OPENDARTS_MAX_DIMS}').")
endif()
message(STATUS "  OBL max dimensions / engine max NC: ${OPENDARTS_MAX_DIMS}")

# Engines library reads MAX_NC from globals.h (now #ifndef-guarded). Forward
# the cmake variable globally so every engine TU and engine pybind exposer
# sees a consistent value. Interpolators target picks up MAX_DIMS via its own
# target_compile_definitions in interpolation/pybind11/CMakeLists.txt.
add_compile_definitions(MAX_NC=${OPENDARTS_MAX_DIMS})
# Cap the mechanical engines' compositional dimensionality independently of
# OPENDARTS_MAX_DIMS. Super-elastic blocks reach N_VARS = NC + 1 + ND, which at
# the global MAX_NC requires linsolv_bos_fs_cpr<N> template instantiations the
# proprietary darts-linear-solvers archive doesn't carry. Fixing this at NC=3
# keeps mech blocks within the existing fs_cpr<4..12> instantiation set (thermal
# super-elastic at NC=3 → N_VARS=7) while still letting flow engines run at the
# full OPENDARTS_MAX_DIMS.
add_compile_definitions(MAX_NC_MECH=3)
# ------------------------------------------------------------------------------


# Report options for this build
feature_summary(WHAT ENABLED_FEATURES DISABLED_FEATURES PACKAGES_FOUND)
feature_summary(FILENAME ${CMAKE_CURRENT_BINARY_DIR}/features.log WHAT ALL)
message(STATUS "   The list of features was also saved to file: ${CMAKE_CURRENT_BINARY_DIR}/features.log\n")

message(CHECK_PASS "done!")  # build settings
# ------------------------------------------------------------------------------


# Setup testing ----------------------------------------------------------------

# Report Enable or disable testing via option flag
if(ENABLE_TESTING)
  enable_testing()
endif()
# ------------------------------------------------------------------------------


# Thirdparty libraries ---------------------------------------------------------
# Reports we started looking for thirdparty libraries and initializes the flag
# that checks if all have been found
message(CHECK_START "Importing thirdparty libraries")
unset(thirdparty_missing_components)

if(NOT ONLY_SOLVERS)
  #   thirdparty/pybind11 ------------------------------------------------------
  include(thirdparty/thirdparty_pybind11.cmake)
endif()

#   thirdparty/SuperLU ---------------------------------------------------------
if(NOT DEFINED BOS_SOLVERS_DIR)
  include(thirdparty/thirdparty_superlu.cmake)
endif()

include(thirdparty/thirdparty_hypre.cmake)

#   thirdparty/bos_solvers ------------------------------------------------------
if(BOS_SOLVERS_DIR)
  include(thirdparty/thirdparty_bos_solvers.cmake)
endif()

# Finalize reporting and check if all libraries have been added ----------------
message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------


# Configure GPU build
if(CUDA)
    find_package(CUDAToolkit REQUIRED)

    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found. Enabling CUDA support.")
        cmake_minimum_required (VERSION 3.24)  # 3.24 is needed for architecture native
        # Set architecture BEFORE enable_language(CUDA) so CMake can detect the compiler
        if(DEFINED CUDA_ARCH)
            message(STATUS "Using CUDA architecture(s): ${CUDA_ARCH}")
            set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
        else()
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
        # Use nvcc discovered by find_package(CUDAToolkit) when it is not in PATH
        if(NOT DEFINED CMAKE_CUDA_COMPILER AND DEFINED CUDAToolkit_BIN_DIR)
            find_program(_NVCC_PATH nvcc PATHS "${CUDAToolkit_BIN_DIR}" NO_DEFAULT_PATH)
            if(_NVCC_PATH)
                set(CMAKE_CUDA_COMPILER "${_NVCC_PATH}" CACHE FILEPATH "CUDA compiler")
                message(STATUS "Using CUDA compiler: ${_NVCC_PATH}")
            endif()
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options='-march=native' --expt-relaxed-constexpr -allow-unsupported-compiler")
        # Workaround for CUDA 13.2 + gcc 13 + CCCL ambiguity in cub/block/block_load_to_shared.cuh
        # where unqualified data()/size() calls match both std:: and cuda::std:: overloads for
        # cuda::std::span<char>. We ship a patched copy of the header that fully qualifies those
        # calls, and prepend its directory to the nvcc include search path so it is picked up
        # before the system CCCL one.
        #
        # The override is scoped narrowly so future CUDA releases can pick up upstream fixes:
        #   1. Only engages for the exact CUDA toolkit versions known to carry the regression
        #      (CUDA 13.2.x; 13.0/13.1 don't ship the offending header, 13.3+ is assumed fixed).
        #   2. Only engages if the *system* CCCL header is actually present on this toolkit
        #      (guarding against custom/minimal CUDA installs).
        #   3. Only engages if the vendored patched copy is present (so deleting this directory
        #      disables the workaround with no further changes).
        # See cmake_patches/cccl_override/README.md for the full rationale and references.
        set(_cccl_override_dir "${CMAKE_SOURCE_DIR}/cmake_patches/cccl_override")
        set(_cccl_override_header "${_cccl_override_dir}/cub/block/block_load_to_shared.cuh")
        set(_cccl_system_header "${CUDAToolkit_TARGET_DIR}/include/cccl/cub/block/block_load_to_shared.cuh")
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.2"
           AND CUDAToolkit_VERSION VERSION_LESS "13.3"
           AND EXISTS "${_cccl_system_header}"
           AND EXISTS "${_cccl_override_header}")
            message(STATUS "Applying CCCL workaround for CUDA ${CUDAToolkit_VERSION} block_load_to_shared.cuh ambiguity")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${_cccl_override_dir}")
        endif()
        unset(_cccl_override_dir)
        unset(_cccl_override_header)
        unset(_cccl_system_header)
        add_compile_definitions(WITH_GPU)
    else()
        message(FATAL_ERROR "CUDA requested but not found!")
    endif()
else()
    message(STATUS "CUDA support is disabled.")
endif()


# Build dummy library to contain compiler flags --------------------------------
add_library(opendarts_compiler_flags INTERFACE)

target_compile_features(opendarts_compiler_flags
  INTERFACE
  cxx_std_20
)

target_compile_options(opendarts_compiler_flags
  INTERFACE
    ${opendarts_cxx_compiler_flags}
)

target_link_options(opendarts_compiler_flags
  INTERFACE
    ${opendarts_cxx_linker_flags}
)
# ------------------------------------------------------------------------------


# openDARTS Linear solvers -----------------------------------------------------
if(NOT DEFINED BOS_SOLVERS_DIR)
  add_subdirectory(solvers)
endif()
# ------------------------------------------------------------------------------


# Interpolation ----------------------------------------------------------------
if(NOT ONLY_SOLVERS)
  add_subdirectory(interpolation)
endif()
# ------------------------------------------------------------------------------


# Engines ----------------------------------------------------------------------
if(NOT ONLY_SOLVERS)
  add_subdirectory(engines)
endif()
# ------------------------------------------------------------------------------


# Discretizer ------------------------------------------------------------------
if(NOT ONLY_SOLVERS)
  add_subdirectory(discretizer)
endif()
# ------------------------------------------------------------------------------


# Include the tests ------------------------------------------------------------
if(ENABLE_TESTING)
  add_subdirectory(tests)
endif()
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Bundle the IPhreeqc shared library into our install tree
# ------------------------------------------------------------------------------
if(WITH_PHREEQC)
  set(IPHREEQC_ROOT "${CMAKE_SOURCE_DIR}/thirdparty/install/iphreeqc")
  if(WIN32)
    set(IPHREEQC_LINK   "${IPHREEQC_ROOT}/bin/IPhreeqc.dll")
    set(IPHREEQC_RENAME "IPhreeqc.dll")
  else()
    set(IPHREEQC_RENAME "libIPhreeqc.so")
    # Debian-based distros install to lib/, RHEL to lib64/. Search both.
    find_library(
      IPHREEQC_LINK
      NAMES IPhreeqc iphreeqc
      PATHS "${IPHREEQC_ROOT}"
      PATH_SUFFIXES lib64 lib lib/x86_64-linux-gnu
      NO_DEFAULT_PATH
    )
  endif()

  # make sure the link exists
  if(NOT IPHREEQC_LINK OR IPHREEQC_LINK STREQUAL "IPHREEQC_LINK-NOTFOUND")
    message(FATAL_ERROR
      "IPhreeqc shared library not found under ${IPHREEQC_ROOT} "
      "(checked lib64, lib, lib/x86_64-linux-gnu)")
  endif()

  # resolve symlinks (on Windows this just canonicalizes the path)
  file(REAL_PATH "${IPHREEQC_LINK}" IPHREEQC_REAL_PATH)
  if(IPHREEQC_REAL_PATH STREQUAL "" OR NOT EXISTS "${IPHREEQC_REAL_PATH}")
    message(FATAL_ERROR
      "Failed to resolve real path of ${IPHREEQC_LINK}; got '${IPHREEQC_REAL_PATH}'")
  endif()
  message(STATUS "Installing IPhreeqc real library: ${IPHREEQC_REAL_PATH}")

  # Copy it at install time into the top of the install prefix
  install(FILES "${IPHREEQC_REAL_PATH}"
          DESTINATION "."
          RENAME   "${IPHREEQC_RENAME}"
          COMPONENT Runtime)
endif()
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Bundle standard C++ library used by the compiler, to support running even on old systems
# ------------------------------------------------------------------------------
if(UNIX AND NOT APPLE)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so.6
        OUTPUT_VARIABLE LIBSTDCXX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    install(CODE "
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${LIBSTDCXX} ${CMAKE_INSTALL_PREFIX})
    ")
endif()
# ------------------------------------------------------------------------------
