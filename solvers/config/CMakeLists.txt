# Setup dynamic header and cpp files -------------------------------------------
# For now just an example, should include also the git commit also. For now only the version is included.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/config/src/version.cpp ESCAPE_QUOTES)
# ------------------------------------------------------------------------------

# Build library to contain base configuration ----------------------------------

# This library will be available in the openDARTS project
# and will contain all the common configuration variables to be
# shared over the different components of the project. For now
# it is placed here since this is the only cmake build file.

# Set library files

# Define the source files (private headers should also go here)
set(OD_CONFIG_SRC_FILES
  ${CMAKE_CURRENT_BINARY_DIR}/config/src/version.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/data_types.cpp
)

# Define the public header files
set(OD_CONFIG_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/openDARTS/config/data_types.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/openDARTS/config/version.hpp
)

# Define the config library
add_library(config OBJECT "")

target_sources(config
  PRIVATE
    ${OD_CONFIG_SRC_FILES}
    ${OD_CONFIG_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
)

target_link_libraries(config opendarts_compiler_flags)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(config
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for config
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${OD_CONFIG_HEADER_FILES})
source_group(src FILES ${OD_CONFIG_SRC_FILES})

if(ONLY_SOLVERS)
  # Install the library file of config
  install(TARGETS config)

  # Install the public header files
  # This is now duplicated but once this is added to the main
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
          DESTINATION include
          COMPONENT header_files
          FILES_MATCHING PATTERN "*.hpp")
endif()
# ------------------------------------------------------------------------------
