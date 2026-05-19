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
    ${CMAKE_CURRENT_SOURCE_DIR}/linearSolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mgrStrategy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compositionalFlowStrategy.cpp
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
  ${CMAKE_CURRENT_SOURCE_DIR}/linear_solver_base.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_superlu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_hypre_amg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_hypre_ilu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_amg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_bilu0.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_cpr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_fs_cpr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_gmres.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_registry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_iface_adapter.cpp
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
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linear_solvers_data_types.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/LinearSolver.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/MGRStrategy.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/CompositionalFlowStrategy.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/Types.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/solver_config.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linear_solver.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/solver_registry.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_iface_adapter.hpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_cusparse_ilu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_cusolv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_cpr_gpu.cu
  )
  set(OD_GPU_SOLVERS_HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/gpu_tools.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_bicgstab.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_cusparse_ilu.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_cusolv.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_bos_cpr_gpu.hpp
  )
  # csr_matrix.cpp and the .cpp GPU wrappers must be compiled as CUDA.
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/csr_matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bicgstab.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_cusparse_ilu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_cusolv.cpp
    PROPERTIES LANGUAGE CUDA
  )
  # AMGX is opt-in (-D WITH_AMGX=ON); it is the only GPU solver with a heavy
  # thirdparty dependency.
  if(WITH_AMGX)
    list(APPEND OD_GPU_SOLVERS_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_amgx.cpp)
    list(APPEND OD_GPU_SOLVERS_HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/linsolv_amgx.hpp)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/linsolv_amgx.cpp PROPERTIES LANGUAGE CUDA)
  endif()
endif()
# ------------------------------------------------------------------------------

# Define the unified linear_solvers library (STATIC, will be linked into shared opendarts_solvers)
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
    SuperLU
    cblas
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
    target_link_libraries(linear_solvers PUBLIC amgx)
    target_include_directories(linear_solvers
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/thirdparty/AMGX/include>
    )
  endif()
endif()

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(linear_solvers
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/interpolation/include>
    $<INSTALL_INTERFACE:include>
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
