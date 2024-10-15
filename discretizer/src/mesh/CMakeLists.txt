# Build discretizer_mesh -------------------------------------------------------
# Set library files

# First the source files (private headers should also go here)
set(DISCRETIZER_MESH_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/elem.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh.cpp
)

# Then the public headers
set(DISCRETIZER_MESH_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/elem.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/vtksnapshotter.h
)

# Define the discretizer_mesh library
# This is always a static library since we wish to use it internally only
add_library(discretizer_mesh OBJECT "")

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with 
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(discretizer_mesh PUBLIC OPENDARTS_LINEAR_SOLVERS)
endif()

target_sources(discretizer_mesh
  PRIVATE
    ${DISCRETIZER_MESH_SRC_FILES}
    ${DISCRETIZER_MESH_PUBLIC_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
)

target_link_libraries(discretizer_mesh
  PUBLIC
    opendarts_compiler_flags
    #auxiliary
    #engines_global
    discretizer_linalg
  PRIVATE 
	  mshio
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(discretizer_mesh
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for discretizer_mesh
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${DISCRETIZER_MESH_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${DISCRETIZER_MESH_PUBLIC_HEADER_FILES})

# Install the library file of discretizer_mesh
install(TARGETS discretizer_mesh
  CONFIGURATIONS RELEASE    
  DESTINATION lib
)

install(TARGETS discretizer_mesh
  CONFIGURATIONS DEBUG     
  DESTINATION lib
)
# ------------------------------------------------------------------------------