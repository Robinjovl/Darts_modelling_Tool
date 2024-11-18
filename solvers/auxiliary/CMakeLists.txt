
# Build library to contain the auxiliary headers -------------------------------

# This library will be available in the openDARTS project
# and will contain all the auxiliary functions and classes shared over the
# different components of the project.

# Set library files

# Define the source filess (private headers should also go here)
set(OD_AUXILIARY_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/timer_node.cpp
)

# Define the public header files
set(OD_AUXILIARY_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/openDARTS/auxiliary/timer_node.hpp
)

# Define the auxiliary library
add_library(auxiliary OBJECT "")

target_sources(auxiliary
  PRIVATE
    ${OD_AUXILIARY_SRC_FILES}
    ${OD_AUXILIARY_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
)

target_link_libraries(auxiliary opendarts_compiler_flags)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(auxiliary
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for auxiliary
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${OD_AUXILIARY_HEADER_FILES})
source_group(src FILES ${OD_AUXILIARY_SRC_FILES})

if(ONLY_SOLVERS)
  # Install the library file of auxiliary
  install(TARGETS auxiliary)

  # Install the public header files
  # This is now duplicated but once this is added to the main
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
          DESTINATION include
          COMPONENT header_files
          FILES_MATCHING PATTERN "*.hpp")
endif()
# ------------------------------------------------------------------------------
