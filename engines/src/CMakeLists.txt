# Setup dynamic header and cpp files -------------------------------------------
# For now just an example, should include also the git commit also. For now only the version is included.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/engines_build_info.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/engines/src/engines_build_info.cpp ESCAPE_QUOTES)

# Build opendarts_engines ------------------------------------------------------

# opendarts_engines is built by parts:
# - globals
# - interpolation
# - mech
# - conn_mesh
# - wells (including stream)
# - engines
# - python_interface

# --- Build engines_globals ----------------------------------------------------
# Set library files

# First the source files (private headers should also go here)
set(OD_ENGINES_GLOBAL_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/globals.cpp
)

# Then the public headers
set(OD_ENGINES_GLOBAL_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/globals.h
)

# Define the engines_global library
# This is always a static library since we wish to use it internally only
add_library(engines_global OBJECT "")

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with 
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(engines_global PUBLIC OPENDARTS_LINEAR_SOLVERS)
endif()

target_sources(engines_global
  PRIVATE
    ${OD_ENGINES_GLOBAL_SRC_FILES}
    ${OD_ENGINES_GLOBAL_PUBLIC_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
)

target_link_libraries(engines_global
  PUBLIC
    opendarts_compiler_flags
    linear_solvers # gets auxiliary and config via linear_solver. Kept like this for compatibility with old linear solvers
)

if (CUDA)
  target_link_libraries(engines_global
    PUBLIC
      CUDA::cudart
  )
endif()

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(engines_global
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for engines_global
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_GLOBAL_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_GLOBAL_PUBLIC_HEADER_FILES})

# Install the library file of engines_global
install(TARGETS engines_global 
  CONFIGURATIONS RELEASE    
  DESTINATION lib
)

install(TARGETS engines_global
  CONFIGURATIONS DEBUG     
  DESTINATION lib
)

# Install the public header files
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
#         DESTINATION include
#         COMPONENT header_files
#         FILES_MATCHING PATTERN "*.hpp")

# ------------------------------------------------------------------------------


# --- Include interpolators ----------------------------------------------------
add_subdirectory(interpolation)
# ------------------------------------------------------------------------------


# --- Include mechanics --------------------------------------------------------
add_subdirectory(mech)
# ------------------------------------------------------------------------------


# --- Build engines_build_info -------------------------------------------------
# Set library files

# First the source files (private headers should also go here)
set(OD_ENGINES_BUILD_INFO_SRC_FILES
  ${CMAKE_CURRENT_BINARY_DIR}/engines/src/engines_build_info.cpp
)

# Then the public headers
set(OD_ENGINES_BUILD_INFO_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/engines_build_info.h
)

# Define the engines_build_info library
# This is always a static library since we wish to use it internally only
add_library(engines_build_info OBJECT "")

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with 
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(engines_build_info PUBLIC OPENDARTS_LINEAR_SOLVERS)
endif()

target_sources(engines_build_info
  PRIVATE
    ${OD_ENGINES_BUILD_INFO_SRC_FILES}
    ${OD_ENGINES_BUILD_INFO_PUBLIC_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
)

target_link_libraries(engines_build_info
  PUBLIC
    opendarts_compiler_flags
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(engines_build_info
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for engines_build_info
#source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_BUILD_INFO_SRC_FILES})
#source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_BUILD_INFO_PUBLIC_HEADER_FILES})

# Install the library file of engines_build_info
install(TARGETS engines_build_info 
  CONFIGURATIONS RELEASE    
  DESTINATION lib
)

install(TARGETS engines_build_info
  CONFIGURATIONS DEBUG     
  DESTINATION lib
)

# # Install the public header files
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
#         DESTINATION include
#         COMPONENT header_files
#         FILES_MATCHING PATTERN "*.hpp")
# ------------------------------------------------------------------------------


# --- Build engines_conn_mesh --------------------------------------------------
# Set library files

# First the source files (private headers should also go here)
set(OD_ENGINES_CONN_MESH_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/conn_mesh.cpp
)

# Then the public headers
set(OD_ENGINES_CONN_MESH_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/conn_mesh.h
)

# Define the engines_conn_mesh library
# This is always a static library since we wish to use it internally only
add_library(engines_conn_mesh OBJECT "")

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with 
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(engines_conn_mesh PUBLIC OPENDARTS_LINEAR_SOLVERS)
endif()

target_sources(engines_conn_mesh
  PRIVATE
    ${OD_ENGINES_CONN_MESH_SRC_FILES}
    ${OD_ENGINES_CONN_MESH_PUBLIC_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
)

target_link_libraries(engines_conn_mesh
  PUBLIC
    engines_global
    engines_wells
    engines_interpolator
    opendarts_compiler_flags
    linear_solvers
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(engines_conn_mesh
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for engines_conn_mesh
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_CONN_MESH_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_CONN_MESH_PUBLIC_HEADER_FILES})

# Install the library file of engines_conn_mesh
install(TARGETS engines_conn_mesh
  CONFIGURATIONS RELEASE    
  DESTINATION lib
)

install(TARGETS engines_conn_mesh
  CONFIGURATIONS DEBUG     
  DESTINATION lib
)

# # Install the public header files
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
#         DESTINATION include
#         COMPONENT header_files
#         FILES_MATCHING PATTERN "*.hpp")
# ------------------------------------------------------------------------------


# --- Build engines_wells ------------------------------------------------------
# Set library files

# First the source files (private headers should also go here)
set(OD_ENGINES_WELLS_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/ms_well.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/well_controls.cpp
)

# Then the public headers
set(OD_ENGINES_WELLS_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/ms_well.h
  ${CMAKE_CURRENT_SOURCE_DIR}/well_controls.h
)

# Define the engines_wells library
# This is always a static library since we wish to use it internally only
add_library(engines_wells OBJECT "")

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with 
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(engines_wells PUBLIC OPENDARTS_LINEAR_SOLVERS)
endif()

target_sources(engines_wells
  PRIVATE
    ${OD_ENGINES_WELLS_SRC_FILES}
    ${OD_ENGINES_WELLS_PUBLIC_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
)

target_link_libraries(engines_wells
  PUBLIC
    opendarts_compiler_flags
    linear_solvers
    engines_interpolator
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(engines_wells
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for engines_wells
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_WELLS_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_WELLS_PUBLIC_HEADER_FILES})

# Install the library file of engines_wells
install(TARGETS engines_wells 
  CONFIGURATIONS RELEASE    
  DESTINATION lib
)

install(TARGETS engines_wells
  CONFIGURATIONS DEBUG     
  DESTINATION lib
)

# Install the public header files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
        DESTINATION include
        COMPONENT header_files
        FILES_MATCHING PATTERN "*.hpp")
# ------------------------------------------------------------------------------


# --- Build engines_engines ----------------------------------------------------
# Set library files

# First the source files (private headers should also go here)
set(OD_ENGINES_ENGINES_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_base.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_nc_cg_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_nce_g_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_nc_nl_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/logging.cpp
)


# Then the public headers
set(OD_ENGINES_ENGINES_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_base.h
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_nc_cg_cpu.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_nce_g_cpu.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_nc_nl_cpu.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_super_cpu.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_super_cpu.tpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_super_mp_cpu.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_super_mp_cpu.tpp
  ${CMAKE_CURRENT_SOURCE_DIR}/logging.h
)

# Define the engines_engines library
# This is always a static library since we wish to use it internally only
add_library(engines_engines OBJECT "")

if (CUDA)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_base_gpu.cpp 

    PROPERTIES LANGUAGE CUDA)

  target_sources(engines_engines PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_base_gpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_nc_cg_gpu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_nce_g_gpu.cu
  )
endif()


# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with 
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(engines_engines PUBLIC OPENDARTS_LINEAR_SOLVERS)
endif()

target_sources(engines_engines
  PRIVATE
    ${OD_ENGINES_ENGINES_SRC_FILES}
    ${OD_ENGINES_ENGINES_PUBLIC_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
	  ${HYPRE_INCLUDE_DIRS}
)

target_link_libraries(engines_engines
  PUBLIC
    opendarts_compiler_flags
    linear_solvers
    engines_interpolator
    Eigen3::Eigen
	HYPRE::HYPRE
    pybind11::pybind11
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(engines_engines
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for engines_engines
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_ENGINES_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_ENGINES_PUBLIC_HEADER_FILES})

# Install the library file of engines_engines
install(TARGETS engines_engines 
  CONFIGURATIONS RELEASE    
  DESTINATION lib
)

install(TARGETS engines_engines
  CONFIGURATIONS DEBUG     
  DESTINATION lib
)

# # Install the public header files
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
#         DESTINATION include
#         COMPONENT header_files
#         FILES_MATCHING PATTERN "*.hpp")
# ------------------------------------------------------------------------------


# Include python interface -----------------------------------------------------
add_subdirectory(pybind11)
# ------------------------------------------------------------------------------
