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
  ${CMAKE_CURRENT_SOURCE_DIR}/linear_solver_base.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_superlu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_hypre_amg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_hypre_ilu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_amg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_bilu0.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_cpr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_fs_cpr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linsolv_bos_gmres.cpp
)

# Public headers
set(OD_LINEAR_SOLVERS_PUBLIC_HEADER_FILES
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
)

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
