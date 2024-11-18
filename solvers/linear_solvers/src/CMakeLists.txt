# Build libopendarts_linear_solvers --------------------------------------------

# Set library files

# First the source files (private headers should also go here)
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

# Then the public headers
set(OD_LINEAR_SOLVERS_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/csr_matrix.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/csr_matrix_base.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/linsolv_iface.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/linsolv_iface_bos.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/linear_solver_base.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/linear_solver_prop.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/linsolv_superlu.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/linsolv_hypre_amg.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/linsolv_hypre_ilu.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/linsolv_bos_amg.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/linsolv_bos_bilu0.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/linsolv_bos_cpr.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/linsolv_bos_fs_cpr.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/linsolv_bos_gmres.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/openDARTS/linear_solvers/data_types.hpp
)

# Define the opendarts_linear_solvers library
# This is always a static library since we wish to use it internally only
add_library(linear_solvers STATIC "")

target_sources(linear_solvers
  PRIVATE
    ${OD_LINEAR_SOLVERS_SRC_FILES}
    ${OD_LINEAR_SOLVERS_PUBLIC_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
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
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for opendarts_linear_solvers
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_LINEAR_SOLVERS_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_LINEAR_SOLVERS_PUBLIC_HEADER_FILES})

if(ONLY_SOLVERS)
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

  # These steps setup a single file 
  # with all dependent libraries included in the same file 
  # In the future with all components of opendarts integrated with cmake 
  # this is no longer needed, for now it is done to simplify integration 
  # with engines.

  # Generate the single library file and determine the include header files directories 
  # This function takes linear_solvers library checks all dependencies, bundle them 
  # all together and places the single .a file into <CMAKE_BINARY_DIR>/bundled_libs.
  # Additionally it returns:
  #   linear_solvers_bundled_filename: the full path to the single. a file
  #   linear_solvers_bundled_header_files: the string containing all header file include drectories
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
