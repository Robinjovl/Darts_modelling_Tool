# Generate config sources -------------------------------------------------------
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/version.cpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/config/src/version.cpp
               ESCAPE_QUOTES)
# ------------------------------------------------------------------------------

# Build config library ----------------------------------------------------------
set(OD_CONFIG_SRC_FILES
  ${CMAKE_CURRENT_BINARY_DIR}/config/src/version.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_types.cpp
)

set(OD_CONFIG_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/data_types.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/version.hpp
)

add_library(config OBJECT "")

if(NOT WIN32)
  set_target_properties(config PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_sources(config
  PRIVATE
    ${OD_CONFIG_SRC_FILES}
    ${OD_CONFIG_HEADER_FILES}
)

target_link_libraries(config opendarts_compiler_flags)

target_include_directories(config
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_CONFIG_HEADER_FILES})
source_group(src FILES ${OD_CONFIG_SRC_FILES})
# ------------------------------------------------------------------------------

# Build auxiliary library -------------------------------------------------------
set(OD_AUXILIARY_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/timer_node.cpp
)

set(OD_AUXILIARY_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/timer_node.hpp
)

add_library(auxiliary OBJECT "")

if(NOT WIN32)
  set_target_properties(auxiliary PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_sources(auxiliary
  PRIVATE
    ${OD_AUXILIARY_SRC_FILES}
    ${OD_AUXILIARY_HEADER_FILES}
)

target_link_libraries(auxiliary opendarts_compiler_flags)

target_include_directories(auxiliary
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_AUXILIARY_HEADER_FILES})
source_group(src FILES ${OD_AUXILIARY_SRC_FILES})
# ------------------------------------------------------------------------------

# ========================================================================
# Build libopendarts_linear_solvers - SINGLE UNIFIED LIBRARY
# ========================================================================
# All solver code goes into one library to avoid ODR violations and
# HYPRE static state issues from having multiple copies of the same code.

# MGR Linear Solver sources
set(OD_MGR_SOLVER_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/mgr_linear_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mgrStrategy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mgr_compositional_flow_strategy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_mgr.cpp
)

# Standard solver sources
set(OD_LINEAR_SOLVERS_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/csr_matrix.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csr_matrix_base.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sparsity_pattern.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csr_expansion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/block_csr_matrix.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scalar_csr_adapter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/block_csr_io.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_bsr_spmv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linear_solver_base.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_superlu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_hypre_amg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_hypre_ilu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_amg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_bilu0.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_cpr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_fs_cpr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_gmres.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_gmres.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_cpr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matrix_slice.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_fs_cpr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_schur_elim.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_registry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_factories.cpp
)

# Public headers
set(OD_LINEAR_SOLVERS_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/dual_array.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/sparsity_pattern.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/csr_expansion.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/block_csr_matrix.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/block_csr_view.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/scalar_csr_adapter.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/block_csr_io.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/gpu_bsr_spmv.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/csr_matrix.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/csr_matrix_base.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_iface.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_iface_bos.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linear_solver_base.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linear_solver_prop.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_mgr.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_superlu.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_hypre_amg.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_hypre_ilu.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_bos_amg.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_bos_bilu0.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_bos_cpr.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_bos_fs_cpr.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_bos_gmres.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_gmres.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_cpr.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/matrix_slice.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_fs_cpr.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_schur_elim.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linear_solvers_data_types.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/mgr_linear_solver.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/MGRStrategy.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/mgr_compositional_flow_strategy.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/Types.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/solver_config.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linear_solver.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/solver_registry.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/solver_configs.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/solver_factories.hpp
)

# GPU solver sources -----------------------------------------------------------
# Only compiled for a CUDA configuration (OPENDARTS_CONFIG=GPU). The .cu files
# carry CUDA kernels; the .cpp wrappers and csr_matrix.cpp issue CUDA runtime /
# cuSPARSE / cuBLAS / cuSOLVER calls and are therefore compiled by nvcc too.
set(OD_GPU_SOLVERS_SRC_FILES "")
set(OD_GPU_SOLVERS_HEADER_FILES "")
if(CUDA)
  set(OD_GPU_SOLVERS_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_tools.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bicgstab.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_gmres_gpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_cusparse_ilu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_cusolv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_cpr_gpu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_mcsgs.cu
  )
  set(OD_GPU_SOLVERS_HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/gpu_tools.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_bicgstab.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_gmres_gpu.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_cusparse_ilu.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_cusolv.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_bos_cpr_gpu.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_mcsgs.hpp
  )
  # csr_matrix.cpp and the .cpp GPU wrappers must be compiled as CUDA.
  # linsolv_schur_elim.cpp carries CUDA kernels behind WITH_GPU and is listed in
  # the always-built sources; in a GPU configuration it must go through nvcc too.
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/csr_matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bicgstab.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_gmres_gpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_cusparse_ilu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_cusolv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_schur_elim.cpp
    PROPERTIES LANGUAGE CUDA
  )
  # AMGX is opt-in (-D WITH_AMGX=ON); it is the only GPU solver with a heavy
  # thirdparty dependency.
  if(WITH_AMGX)
    list(APPEND OD_GPU_SOLVERS_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_amgx.cpp)
    list(APPEND OD_GPU_SOLVERS_HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_amgx.hpp)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/linsolv_amgx.cpp PROPERTIES LANGUAGE CUDA)
  endif()
  # cuDSS is opt-in (-D WITH_CUDSS=ON); prebuilt NVIDIA library, imported via
  # thirdparty/thirdparty_cudss.cmake as cudss::cudss.
  if(WITH_CUDSS)
    list(APPEND OD_GPU_SOLVERS_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_cudss.cpp)
    list(APPEND OD_GPU_SOLVERS_HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_cudss.hpp)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/linsolv_cudss.cpp PROPERTIES LANGUAGE CUDA)
  endif()
endif()
# ------------------------------------------------------------------------------

# Auto-generated explicit template instantiations
# ------------------------------------------------------------------------------
# Every block-size-templated class in solvers/src drives its
# `template class <CLASS><N>;` instantiation set from CMake via
# od_emit_template_instantiations(), instead of hand-maintained walls in the .cpp.
# The .cpp files #include the generated *_instantiations.inc from inside their
# namespace. Edit the ranges here, not in the sources.
#
# (The only linear solver NOT covered here is the external prebuilt package
# consumed via -b / ENABLE_BOS_SOLVERS, which is not part of this repository.)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/od_emit_template_instantiations.cmake)

# Shared block-size range for the per-N flow solvers. The +1 covers thermal
# super-engine blocks (N_VARS = NC + 1); the floor of 13 keeps the range stable
# (and satisfies the solver unit-test contract) regardless of OPENDARTS_MAX_DIMS
# (default 8 would otherwise leave N=10..13 unbound).
math(EXPR _od_linsolv_max_n "${OPENDARTS_MAX_DIMS} + 1")
if(_od_linsolv_max_n LESS 13)
  set(_od_linsolv_max_n 13)
endif()
set(_od_linsolv_gen_dir ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${_od_linsolv_gen_dir})

# Solvers/containers whose smallest meaningful block size is 1; max tracks
# max(OPENDARTS_MAX_DIMS+1, 13). Includes the CUDA-only solvers (their .inc is
# generated unconditionally; it is only #included when the TU is compiled), the
# in-tree SuperLU direct solver (thirdparty/superlu submodule, built by default),
# the block-CSR matrix container, and the default block preconditioners/solvers.
foreach(_od_solver linsolv_gmres linsolv_cpr linsolv_mgr linsolv_superlu
                   linsolv_bicgstab linsolv_gmres_gpu linsolv_cusolv
                   linsolv_cusparse_ilu linsolv_cudss linsolv_amgx linsolv_mcsgs
                   linsolv_bos_bilu0 linsolv_bos_gmres csr_matrix)
  od_emit_template_instantiations(
    CLASS ${_od_solver}
    MAX_N ${_od_linsolv_max_n}
    OUTPUT_FILE ${_od_linsolv_gen_dir}/${_od_solver}_instantiations.inc
  )
endforeach()

# The mineral-elimination wrapper is a two-parameter template <N, K>; its
# explicit instantiations live directly in linsolv_schur_elim.cpp (the single-
# parameter emitter cannot express the second parameter). Pass the same block-
# size ceiling as a compile define so its instantiation range stays in lock-step
# with the per-N solvers (and thus covers every engine block size, N_VARS up to
# MAX_NC+1) instead of a hardcoded 13 that would link-fail for OPENDARTS_MAX_DIMS>=13.
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_schur_elim.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_factories.cpp
  PROPERTIES COMPILE_DEFINITIONS "OD_SE_NMAX=${_od_linsolv_max_n}")

# CPR solvers start at block size 2 — the constrained-pressure-residual split is
# undefined for a 1x1 block (no non-pressure sub-block to decouple).
foreach(_od_solver linsolv_bos_cpr linsolv_bos_cpr_gpu)
  od_emit_template_instantiations(
    CLASS ${_od_solver}
    MIN_N 2
    MAX_N ${_od_linsolv_max_n}
    OUTPUT_FILE ${_od_linsolv_gen_dir}/${_od_solver}_instantiations.inc
  )
endforeach()

# Full-system CPR (super-elastic mechanics) uses a fixed block-size window
# N_VARS = NC + 1 + ND, i.e. 4..8 — deliberately NOT coupled to OPENDARTS_MAX_DIMS.
od_emit_template_instantiations(
  CLASS linsolv_fs_cpr
  MIN_N 4
  MAX_N 8
  OUTPUT_FILE ${_od_linsolv_gen_dir}/linsolv_fs_cpr_instantiations.inc
)

# Compatibility solvers with fixed, domain-specific block sizes (not coupled to
# OPENDARTS_MAX_DIMS): the default full-system CPR (block size 4 only) and the
# HYPRE scalar preconditioners (block size 1 — the reduced pressure system).
od_emit_template_instantiations(
  CLASS linsolv_bos_fs_cpr MIN_N 4 MAX_N 4
  OUTPUT_FILE ${_od_linsolv_gen_dir}/linsolv_bos_fs_cpr_instantiations.inc
)
od_emit_template_instantiations(
  CLASS linsolv_hypre_amg MIN_N 1 MAX_N 1
  OUTPUT_FILE ${_od_linsolv_gen_dir}/linsolv_hypre_amg_instantiations.inc
)
od_emit_template_instantiations(
  CLASS linsolv_hypre_ilu MIN_N 1 MAX_N 1
  OUTPUT_FILE ${_od_linsolv_gen_dir}/linsolv_hypre_ilu_instantiations.inc
)

# Default AMG supports a non-contiguous block-size set {1, 3}; block size 2 is
# intentionally absent (see the VALUES option of od_emit_template_instantiations).
od_emit_template_instantiations(
  CLASS linsolv_bos_amg VALUES "1;3"
  OUTPUT_FILE ${_od_linsolv_gen_dir}/linsolv_bos_amg_instantiations.inc
)
# ------------------------------------------------------------------------------

# Define the unified linear_solvers library (STATIC, will be linked into shared opendarts_linear_solvers)
add_library(linear_solvers STATIC "")

# Build as PIC so it can link into shared objects (Python extensions, etc.)
if(NOT WIN32)
    set_target_properties(linear_solvers PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_sources(linear_solvers
  PRIVATE
    ${OD_MGR_SOLVER_SRC_FILES}
    ${OD_LINEAR_SOLVERS_SRC_FILES}
    ${OD_LINEAR_SOLVERS_PUBLIC_HEADER_FILES}
    ${OD_GPU_SOLVERS_SRC_FILES}
    ${OD_GPU_SOLVERS_HEADER_FILES}
)

target_link_libraries(linear_solvers
  PUBLIC
    opendarts_compiler_flags
    config
    auxiliary
    superlu::superlu
    HYPRE::HYPRE
)

# Link the CUDA libraries used by the GPU solver wrappers.
if(CUDA)
  target_link_libraries(linear_solvers
    PUBLIC
      CUDA::cudart
      CUDA::cusparse
      CUDA::cublas
      CUDA::cusolver
  )
  if(WITH_AMGX)
    # AMGX builds a static (amgx) and a shared (amgxsh) target; link the
    # static one so no extra runtime .so has to be shipped.
    #
    # PRIVATE on purpose: the amgx target's INTERFACE carries AMGX-internal
    # compile flags (-DTHRUST_CUB_WRAPPED_NAMESPACE=amgx, -Werror
    # cross-execution-space-call, its own -Xcompiler block, ...). Linking
    # PUBLIC leaks them into every consumer's CUDA TU, wrapping Thrust into
    # the amgx:: namespace and breaking plain thrust:: usage in
    # interpolation/ and engines/. linsolv_amgx.hpp does not expose any AMGX
    # header (implementation-only dependency), so consumers need neither the
    # flags nor the include dir; CMake still propagates the static-link
    # requirement itself for PRIVATE deps of a static library.
    target_link_libraries(linear_solvers PRIVATE amgx)
    target_include_directories(linear_solvers
      PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/thirdparty/AMGX/include>
    )
  endif()
  if(WITH_CUDSS)
    # Prebuilt shared library; the imported target carries its include dir.
    target_link_libraries(linear_solvers PUBLIC cudss::cudss)
  endif()
endif()

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(linear_solvers
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/interpolation/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    # Auto-generated *_instantiations.inc live here (od_emit_template_instantiations).
    $<BUILD_INTERFACE:${_od_linsolv_gen_dir}>
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(linear_solvers PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Setup IDE directory structure
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_LINEAR_SOLVERS_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_MGR_SOLVER_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_LINEAR_SOLVERS_PUBLIC_HEADER_FILES})

message(STATUS "Unified linear_solvers library configured (includes MGR)")

# ========================================================================
# Bundled library for standalone distribution (ONLY_SOLVERS mode)
# ========================================================================

if(ONLY_SOLVERS)
  # Install config and auxiliary object libraries
  install(TARGETS config auxiliary)

  # Install the library file of opendarts_linear_solvers
  install(TARGETS linear_solvers
    CONFIGURATIONS RELEASE
    DESTINATION lib/unbundled
  )

  install(TARGETS linear_solvers
    CONFIGURATIONS DEBUG
    DESTINATION lib/unbundled/debug
  )

  # Install the public header files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
          DESTINATION include
          COMPONENT header_files
          FILES_MATCHING PATTERN "*.hpp")

  # Setup the bundled linear_solvers library
  # Single file for delivery
  set(bundled_tgt_file_name "darts_linear_solvers")
  bundle_static_library(linear_solvers "bundled_libs" linear_solvers_bundled_filename "${bundled_tgt_file_name}" linear_solvers_bundled_header_files linear_solvers_bundling_target)

  # Generate the single file library as an imported library
  add_library(linear_solvers_bundled STATIC IMPORTED GLOBAL)
  set_target_properties(linear_solvers_bundled
    PROPERTIES
    IMPORTED_LOCATION ${linear_solvers_bundled_filename}
    INTERFACE_INCLUDE_DIRECTORIES "${linear_solvers_bundled_header_files}"
  )

  add_dependencies(linear_solvers_bundled linear_solvers_bundling_target)

  message(STATUS "-----------------------------------------------")

  install(FILES ${linear_solvers_bundled_filename}
    CONFIGURATIONS RELEASE RELWITHDEBINFO ""
    DESTINATION lib
  )

  install(FILES ${linear_solvers_bundled_filename}
    CONFIGURATIONS DEBUG
    DESTINATION lib
    RENAME ${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_tgt_file_name}_d${CMAKE_STATIC_LIBRARY_SUFFIX}
  )
endif()
# ------------------------------------------------------------------------------
