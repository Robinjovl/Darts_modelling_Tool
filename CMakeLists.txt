cmake_minimum_required (VERSION 3.26)
# C is enabled (in addition to CXX) so find_package(OpenMP) provides the
# OpenMP::OpenMP_C imported target that an OpenMP-enabled HYPRE (a C library,
# built with HYPRE_OPENMP=1) links against. No first-party C sources are
# compiled; this only makes the C toolchain + OpenMP_C component available.
project (openDARTS LANGUAGES C CXX VERSION 0.2)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure runtime path for finding dynamic libraries.
# $ORIGIN         -- a module resolves siblings in its own directory.
# $ORIGIN/linear_solvers -- the compiled solvers live in the darts/linear_solvers/ subpackage,
#                    and so do the shared libs bundled next to them (libcudss.so.0,
#                    libopendarts_linear_solvers.so). A darts/ top-level extension such as
#                    engines links the solvers statically and thereby inherits a
#                    direct NEEDED on the *shared* libcudss.so.0 (cuDSS is linked
#                    PUBLIC); engines sits in darts/, so plain $ORIGIN cannot reach
#                    darts/linear_solvers/. Adding $ORIGIN/linear_solvers lets it (WITH_CUDSS GPU
#                    builds otherwise fail at `import darts.engines` with
#                    "libcudss.so.0: cannot open shared object file"). Harmless for
#                    non-GPU builds (the extra entry just points at darts/linear_solvers/).
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/linear_solvers")
set(CMAKE_MACOSX_RPATH 1)

# Set DARTS installation folder
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/darts/")

# Disable C++ 11 ABI to support old libraries
set(SET_CXX11_ABI_0 TRUE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # useful for code tools like clangd or IDEs

# Turn off valgrind support by default
option(ENABLE_VALGRIND "Enable Valgrind instrumentation (Debug build with symbols)" OFF)

# Turn off ThreadSanitizer support by default
option(ENABLE_TSAN "Enable ThreadSanitizer instrumentation (-fsanitize=thread)" OFF)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    if(ENABLE_VALGRIND)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Default build type" FORCE)
    elseif(ENABLE_TSAN)
        # RelWithDebInfo, not Debug: TSan already runs the program 5-15x slower
        # than native; -O0 on top of that makes even small models impractically
        # slow, and -g here already gives usable stack traces.
        set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Default build type" FORCE)
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

if(ENABLE_TSAN)
    message(STATUS ">>> ThreadSanitizer build: adding -fsanitize=thread")
    # -fno-omit-frame-pointer: TSan's stack unwinder needs real frame pointers
    # for readable race reports. Applied to CMAKE_*_FLAGS (not a _DEBUG/
    # _RELWITHDEBINFO variant) so it takes effect regardless of build type.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fsanitize=thread -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread")
    # Compile-time ignorelist for code proven (via a from-scratch minimal
    # reproduction: a bare #pragma omp parallel region writing into a
    # per-thread stack slot, read back on the main thread right after the
    # region closes, flags under both GCC's libgomp and a standard LLVM
    # libomp with no application code involved at all) to only ever trigger a
    # known ThreadSanitizer+OpenMP false positive, never a real race. Skips
    # instrumentation entirely for these functions rather than just hiding the
    # report, so it costs nothing at runtime either.
    # Clang-only: GCC has no file/function-pattern sanitizer ignorelist flag.
    # The CI build uses GCC, so this list is enforced there via the runtime
    # suppressions file (helper_scripts/tsan.supp) instead -- see that file
    # for the matching entries.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(TSAN_IGNORELIST "${CMAKE_SOURCE_DIR}/helper_scripts/tsan-ignorelist.list")
        if(EXISTS "${TSAN_IGNORELIST}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize-ignorelist=${TSAN_IGNORELIST}")
            set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -fsanitize-ignorelist=${TSAN_IGNORELIST}")
        endif()
    endif()
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

# Proprietary BOS solvers linkage -----------------------------------------------
# ENABLE_BOS_SOLVERS is the single switch between the in-tree open-source
# solver stack (OFF, default -- darts.solvers registry, FGMRES+CPR CPU
# default) and the proprietary darts-linear-solvers (`bos`) stack (ON --
# requires BOS_SOLVERS_DIR pointing at its lib/include tree; the C++ default
# CPU solver then comes from the engine factory: CPU_GMRES_CPR_AMG for flow
# engines, CPU_GMRES_FS_CPR for mechanics engines). Every CMake guard that
# used to test `DEFINED BOS_SOLVERS_DIR` now tests this option, so the
# OPENDARTS_LINEAR_SOLVERS compile macro tracks exactly NOT ENABLE_BOS_SOLVERS.
option(ENABLE_BOS_SOLVERS "Link the proprietary BOS solvers instead of the in-tree open-source stack" OFF)
add_feature_info(ENABLE_BOS_SOLVERS ENABLE_BOS_SOLVERS "Link the proprietary BOS solvers")
# Back-compat: an explicit BOS_SOLVERS_DIR used to be the switch; keep old
# invocations working by implying ENABLE_BOS_SOLVERS=ON -- but only ONCE per
# build directory. Previously the shim re-fired on every configure, so an
# explicit -D ENABLE_BOS_SOLVERS=OFF could never win while BOS_SOLVERS_DIR
# remained in the cache.
if(DEFINED BOS_SOLVERS_DIR AND NOT ENABLE_BOS_SOLVERS)
  if(NOT DEFINED _BOS_SHIM_APPLIED)
    message(STATUS "  BOS_SOLVERS_DIR given -> ENABLE_BOS_SOLVERS=ON (back-compat)")
    set(ENABLE_BOS_SOLVERS ON)
    set(ENABLE_BOS_SOLVERS ON CACHE BOOL
        "Link the proprietary BOS solvers instead of the in-tree open-source stack" FORCE)
  else()
    message(STATUS
        "  BOS_SOLVERS_DIR is cached but ENABLE_BOS_SOLVERS=OFF -- building the "
        "open-source stack (pass -D ENABLE_BOS_SOLVERS=ON to link BOS).")
  endif()
endif()
set(_BOS_SHIM_APPLIED ON CACHE INTERNAL "back-compat BOS shim evaluated")
if(ENABLE_BOS_SOLVERS AND NOT DEFINED BOS_SOLVERS_DIR)
  message(FATAL_ERROR "ENABLE_BOS_SOLVERS=ON requires -D BOS_SOLVERS_DIR=<path to darts-linear-solvers>")
endif()
# --------------------------------------------------------------------------------

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
  # MT and GPU both build against the in-tree open-source solvers by default.
  # The in-tree block_csr_matrix now provides a real multi-threaded row
  # partition (see linear_solvers/include/omp_partition.hpp), so an MT build no longer
  # depends on the proprietary multi-threaded bos_solvers. Pass
  # -D BOS_SOLVERS_DIR=<path> to link the proprietary bos_solvers instead.
  if (GPU_CONFIG)
    set(CUDA 1)
  endif()
  if(NOT ENABLE_BOS_SOLVERS)
    message(STATUS "  openDARTS ${OPENDARTS_CONFIG} build using the in-tree open-source solvers (no BOS_SOLVERS_DIR)")
  endif()
else()
  message(FATAL_ERROR "openDARTS ${OPENDARTS_CONFIG} is an invalid configuration use: ST, MT or GPU.")
endif()

message(STATUS "  Building openDARTS ${OPENDARTS_CONFIG}")

# Determine the type of compiler we have and set compiler flags and other compiler
# specific flags
if(ENABLE_BOS_SOLVERS)
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
  set(opendarts_cxx_compiler_flags "/MP;/std:c++20;/utf-8")

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

  if(${OPENDARTS_CONFIG} STREQUAL "MT" OR GPU_CONFIG)
    # Host-side OpenMP for both the MT and the GPU build. The GPU build's
    # host code (engine assembly partition, in-tree GMRES kernels, HYPRE
    # consumers) is the same OpenMP code the MT build runs; GPU TUs compiled
    # by nvcc need the flag forwarded to the host compiler -- raw -fopenmp is
    # not an nvcc option, so pass the -Xcompiler form for CUDA sources.
    list(APPEND opendarts_cxx_compiler_flags "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fopenmp>")
    list(APPEND opendarts_cxx_compiler_flags "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>")
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

  if(${OPENDARTS_CONFIG} STREQUAL "MT" OR GPU_CONFIG)
    # Same MT-or-GPU OpenMP policy as the GNU block above.
    list(APPEND opendarts_cxx_compiler_flags "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fopenmp>")
    list(APPEND opendarts_cxx_compiler_flags "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>")
    list(APPEND opendarts_cxx_linker_flags "-fopenmp")
  endif()

  if(SET_CXX11_ABI_0)
    list(APPEND opendarts_cxx_compiler_flags "-D_GLIBCXX_USE_CXX11_ABI=0")
  endif()

  # Compiler specific helper variables (currently not needed)

endif()

# macOS linker hygiene ---------------------------------------------------------
# The Xcode 15+ "ld_prime" linker warns about libraries appearing more than once
# on a link line ("ignoring duplicate libraries"). open-DARTS work item #52
# suggested forcing the classic linker with -ld_classic, but that flag is now
# DEPRECATED on Xcode 16+ (it warns and is slated for removal), so we do NOT
# force it. The portable fix is to suppress the duplicate-library warning via the
# CMake LINKER: prefix (-> -Wl,/-Xlinker as appropriate), gated to Apple only.
# (Collapsing SuperLU+cblas into the single superlu::superlu target already
# removes the main duplicate-archive source.) If an older-Xcode escape hatch is
# ever needed, add an opt-in cache var rather than defaulting -ld_classic on.
# NOTE: untestable from the Linux/RHEL dev box -- correct-by-construction; must
# be validated on a macOS (Apple Silicon) runner.
if(APPLE)
  add_link_options("LINKER:-no_warn_duplicate_libraries")
endif()
# ------------------------------------------------------------------------------

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
# AMGX is the production GPU preconditioner: the default GPU linear solver is
# GPU_GMRES_CPR_AMGX_ILU (engine_base_gpu.h), so a GPU build WITHOUT AMGX
# silently degrades to the far slower BiCGStab+cuSPARSE-ILU(0) fallback. AMGX is
# therefore ON by default for GPU builds. It is compiled from the thirdparty/AMGX
# submodule (open-source, statically linked -- no runtime .so is shipped); the
# GPU build scripts init the submodule automatically. It IS a heavy compile, so
# it can be disabled explicitly with -D WITH_AMGX=OFF (or --no-amgx to the GPU
# build script) for a fast CUDA-toolkit-only build that forgoes the AMGX solver.
option(WITH_AMGX "Build the AMGX GPU solver (thirdparty/AMGX submodule; ON by default for GPU builds)" ON)
add_feature_info(WITH_AMGX WITH_AMGX "Build the AMGX GPU solver")
# cuDSS (NVIDIA sparse direct solver) is a PREBUILT dependency (no source
# build, unlike AMGX). ON by default -- it is only consumed inside the GPU
# (CUDA) branch, so this is effectively "on for GPU builds, ignored on CPU".
# The library is located by thirdparty/thirdparty_cudss.cmake (pip wheel
# nvidia-cudss-cu13, CUDSS_ROOT, or a CMake config package); if it is not
# found the GPU stack is built WITHOUT cuDSS (warning, not error), so a
# cuDSS-less GPU machine still builds. Disable explicitly with -D WITH_CUDSS=OFF.
option(WITH_CUDSS "Build the cuDSS GPU direct solver (prebuilt NVIDIA library)" ON)
add_feature_info(WITH_CUDSS WITH_CUDSS "Build the cuDSS GPU direct solver")

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

# OPENDARTS_MAX_DIMS = N drives BOTH from one knob (kept equal):
#   - MAX_DIMS (interpolation_config.h): OBL dimensions, N_DIMS in 1..N
#   - MAX_NC   (engines/src/globals.h) : engine components, NC in 2..N
# A model's OBL dimension is NC (isothermal) or NC+1 (thermal, +1 per history axis),
# so MAX_NC = MAX_DIMS = N supports isothermal up to N components and thermal up to N-1.
# Example (default N=8): isothermal up to 8 components, thermal up to 7; thermal at 8
# components needs N_DIMS=9, which is not compiled and would fail -- bump N to 9 for that.
# N is the dominant compile-time/RAM knob (interpolate_with_derivatives unrolls
# O(2^N_DIMS) per instantiation), so raise it only when needed. Default 8 = historical value.
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
if(NOT ENABLE_BOS_SOLVERS)
  include(thirdparty/thirdparty_superlu.cmake)
endif()

include(thirdparty/thirdparty_hypre.cmake)

#   thirdparty/bos_solvers ------------------------------------------------------
if(ENABLE_BOS_SOLVERS)
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
            # Use CMake's include-directory machinery instead of appending a
            # raw ``-I`` token to CMAKE_CUDA_FLAGS. The latter is split by the
            # shell when the checkout path contains spaces, causing nvcc to
            # interpret the path fragments as additional input files.
            include_directories(
              BEFORE
                "$<$<COMPILE_LANGUAGE:CUDA>:${_cccl_override_dir}>"
            )
        endif()
        unset(_cccl_override_dir)
        unset(_cccl_override_header)
        unset(_cccl_system_header)
        add_compile_definitions(WITH_GPU)

        # AMGX GPU solver -- ON by default (it backs the default GPU solver
        # GPU_GMRES_CPR_AMGX_ILU). The GPU build scripts init the thirdparty/AMGX
        # submodule automatically; a direct CMake configure without it fails
        # loudly here (rather than silently degrading the default GPU solver).
        if(WITH_AMGX)
            set(_amgx_dir "${CMAKE_SOURCE_DIR}/thirdparty/AMGX")
            if(NOT EXISTS "${_amgx_dir}/CMakeLists.txt")
                message(FATAL_ERROR "WITH_AMGX is ON (default for GPU builds) but the AMGX "
                    "submodule is missing. Init it with:\n"
                    "    git submodule update --init --recursive thirdparty/AMGX\n"
                    "or, to build the GPU stack WITHOUT the AMGX solver (slower default GPU "
                    "solver), pass -D WITH_AMGX=OFF (or --no-amgx to build_install_darts_gpu.sh).")
            endif()
            message(STATUS "Building AMGX GPU solver support from ${_amgx_dir}")
            # AMGX is an MPI-optional library; openDARTS does not use its MPI path.
            set(AMGX_NO_RPATH ON CACHE BOOL "" FORCE)
            # AMGX targets C++17. Its own CMakeLists forces that only for
            # CUDA >= 13.1 (where Thrust/CCCL require it); under older
            # toolkits (e.g. the CUDA 12.6 tahiti CI runner) it inherits this
            # project's C++20, and GCC >= 11 then floods the build log with
            # the on-by-default -Wdeprecated-enum-enum-conversion warning
            # from AMGX's AMGX_Mode enum arithmetic -- megabytes of output,
            # enough to overflow GitLab's 4 MB job-log limit and hide
            # everything after it. add_subdirectory snapshots the calling
            # scope, so pin the standard for the AMGX subtree only and
            # restore ours afterwards. The extra nvcc flag silences the
            # per-TU "sm_70 support will be removed" notice on Volta CI.
            set(_od_saved_cxx_standard "${CMAKE_CXX_STANDARD}")
            set(_od_saved_cuda_standard "${CMAKE_CUDA_STANDARD}")
            set(_od_saved_cuda_flags "${CMAKE_CUDA_FLAGS}")
            set(CMAKE_CXX_STANDARD 17)
            set(CMAKE_CUDA_STANDARD 17)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
            add_subdirectory(thirdparty/AMGX)
            set(CMAKE_CXX_STANDARD "${_od_saved_cxx_standard}")
            set(CMAKE_CUDA_STANDARD "${_od_saved_cuda_standard}")
            set(CMAKE_CUDA_FLAGS "${_od_saved_cuda_flags}")
            unset(_od_saved_cxx_standard)
            unset(_od_saved_cuda_standard)
            unset(_od_saved_cuda_flags)
            add_compile_definitions(WITH_AMGX)
            unset(_amgx_dir)
        endif()

        # cuDSS GPU direct solver (prebuilt library -- imported target, no source
        # build). ON by default for GPU builds; thirdparty/thirdparty_cudss.cmake
        # sets WITH_CUDSS=OFF (FORCE) with a warning if the prebuilt library is not
        # found, so a cuDSS-less GPU build still succeeds. Re-check WITH_CUDSS after
        # the include before defining the compile macro / pulling in the source.
        if(WITH_CUDSS)
            include(thirdparty/thirdparty_cudss.cmake)
        endif()
        if(WITH_CUDSS)
            add_compile_definitions(WITH_CUDSS)
        endif()
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

# In a CUDA build WITH_GPU is defined globally, so headers guarded by it
# (e.g. timer_node.hpp -> <cuda_runtime.h>) are pulled into ordinary
# host-compiled translation units too. Put the CUDA toolkit headers on the
# include path of every target so those includes resolve under g++ as well
# as nvcc. (The cudart link itself is provided where the symbols are used.)
if(CUDA)
  target_include_directories(opendarts_compiler_flags INTERFACE ${CUDAToolkit_INCLUDE_DIRS})

  # CMake obtains cudadevrt/cudart_static from nvcc's implicit link line. With
  # a toolkit installed below a path containing spaces, CUDA 13.2's verbose
  # output can leave CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES truncated at the
  # first space. Add the already-discovered runtime directory explicitly so
  # ordinary C++ link steps can resolve those implicit ``-l`` entries.
  if(CUDAToolkit_LIBRARY_DIR)
    target_link_directories(
      opendarts_compiler_flags
      BEFORE
      INTERFACE
        "${CUDAToolkit_LIBRARY_DIR}"
    )
  endif()
endif()
# ------------------------------------------------------------------------------


# openDARTS Linear solvers -----------------------------------------------------
if(NOT ENABLE_BOS_SOLVERS)
  add_subdirectory(linear_solvers)
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
