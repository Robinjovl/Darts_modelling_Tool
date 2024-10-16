# Setup dynamic header and cpp files -------------------------------------------
# For now just an example, should include also the git commit also. For now only the version is included.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/discretizer_build_info.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/discretizer/src/discretizer_build_info.cpp ESCAPE_QUOTES)

# Build opendarts_discretizer --------------------------------------------------

# opendarts_discretizer is built by parts:
# - Build discretizer_build_info
# - discretizer
# - linalg
# - mesh
# - python_interface

# --- Build discretizer_build_info -------------------------------------------------
# Set library files

# First the source files (private headers should also go here)
set(DISCRETIZER_BUILD_INFO_SRC_FILES
  ${CMAKE_CURRENT_BINARY_DIR}/discretizer/src/discretizer_build_info.cpp
)

# Then the public headers
set(DISCRETIZER_BUILD_INFO_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/discretizer_build_info.h
)

# Define the discretizer_build_info library
# This is always a static library since we wish to use it internally only
add_library(discretizer_build_info OBJECT "")

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with 
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(discretizer_build_info PUBLIC OPENDARTS_LINEAR_SOLVERS)
endif()

target_sources(discretizer_build_info
  PRIVATE
    ${DISCRETIZER_BUILD_INFO_SRC_FILES}
    ${DISCRETIZER_BUILD_INFO_PUBLIC_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
)

target_link_libraries(discretizer_build_info
  PUBLIC
    opendarts_compiler_flags
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(discretizer_build_info
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for discretizer_build_info
#source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${DISCRETIZER_BUILD_INFO_SRC_FILES})
#source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${DISCRETIZER_BUILD_INFO_PUBLIC_HEADER_FILES})

# Install the library file of discretizer_build_info
install(TARGETS discretizer_build_info 
  CONFIGURATIONS RELEASE    
  DESTINATION lib
)

install(TARGETS discretizer_build_info
  CONFIGURATIONS DEBUG     
  DESTINATION lib
)

# # Install the public header files
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
#         DESTINATION include
#         COMPONENT header_files
#         FILES_MATCHING PATTERN "*.hpp")
# ------------------------------------------------------------------------------


# --- Build opendarts_discretizer ----------------------------------------------

# First the source files (private headers should also go here)
set(DISCRETIZER_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/discretizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mech_discretizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)

# Then the public headers
set(DISCRETIZER_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/approximation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/discretizer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mech_discretizer.h  
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
)

# Define the opendarts_discretizer library
# This is always a static library since we wish to use it internally only
add_library(opendarts_discretizer STATIC "")

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with 
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(opendarts_discretizer PUBLIC OPENDARTS_LINEAR_SOLVERS)
endif()

target_sources(opendarts_discretizer
  PRIVATE
    ${DISCRETIZER_SRC_FILES}
    ${DISCRETIZER_PUBLIC_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
)

target_link_libraries(opendarts_discretizer
  PUBLIC
    opendarts_compiler_flags
    discretizer_build_info
    discretizer_mesh
    discretizer_linalg
	  Eigen3::Eigen
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(opendarts_discretizer
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for opendarts_discretizer
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${DISCRETIZER_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${DISCRETIZER_PUBLIC_HEADER_FILES})

# Install the library file of opendarts_discretizer
install(TARGETS opendarts_discretizer 
  CONFIGURATIONS RELEASE    
  DESTINATION lib
)

install(TARGETS opendarts_discretizer
  CONFIGURATIONS DEBUG     
  DESTINATION lib
)

# Install the public header files
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
#         DESTINATION include
#         COMPONENT header_files
#         FILES_MATCHING PATTERN "*.hpp")

# ------------------------------------------------------------------------------


# --- Build linalg (header only) -----------------------------------------------
# First the source files - no source files
# Then the public headers
set(DISCRETIZER_LINALG_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/linalg/matrix.h
  ${CMAKE_CURRENT_SOURCE_DIR}/linalg/vector3.h
)

# Define the discretizer_build_info library
# This is always a static library since we wish to use it internally only
add_library(discretizer_linalg INTERFACE "")

target_sources(discretizer_linalg
  INTERFACE
    ${DISCRETIZER_LINALG_PUBLIC_HEADER_FILES}
)

target_link_libraries(discretizer_linalg
  INTERFACE
    opendarts_compiler_flags
	  Eigen3::Eigen
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(discretizer_linalg
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for discretizer_linalg
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${DISCRETIZER_LINALG_PUBLIC_HEADER_FILES})

# Install the library file of discretizer_linalg
install(TARGETS discretizer_linalg 
  CONFIGURATIONS RELEASE    
  DESTINATION lib
)

install(TARGETS discretizer_linalg
  CONFIGURATIONS DEBUG     
  DESTINATION lib
)

# # Install the public header files
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
#         DESTINATION include
#         COMPONENT header_files
#         FILES_MATCHING PATTERN "*.hpp")
# ------------------------------------------------------------------------------


# --- Build mesh ---------------------------------------------------------------
add_subdirectory(mesh)


# --- Include discretizer python interface -------------------------------------
add_subdirectory(pybind11)
# ------------------------------------------------------------------------------