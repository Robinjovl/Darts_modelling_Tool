
# The python interface for opendarts_engines is done by parts
# - engines_py_globals
# - engines_py_engines
# - engines_py_wells
# - engines_py_interpolators
# - engines_build_info

# Build engines_py_globals -------------------------------------------

# Set library files

# First the source files (private headers should also go here)
set(OD_ENGINES_PY_GLOBALS_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/py_globals.cpp
)


# Then the public headers
set(OD_ENGINES_PY_GLOBALS_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/py_globals.h
)

# Define the engines_py_globals library
# This is always a static library since we wish to use it internally only
add_library(engines_py_globals OBJECT "")

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with 
# the opendarts linear solvers, otherwise it uses the old linear solvers
target_compile_definitions(engines_py_globals 
  PUBLIC 
    OPENDARTS_LINEAR_SOLVERS
    PYBIND11_ENABLED
  )

target_sources(engines_py_globals
  PRIVATE
    ${OD_ENGINES_PY_GLOBALS_SRC_FILES}
    ${OD_ENGINES_PY_GLOBALS_PUBLIC_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
)

target_link_libraries(engines_py_globals
  PUBLIC
    opendarts_compiler_flags
    engines_global
    engines_wells
    engines_build_info
    pybind11::module
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(engines_py_globals
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for engines_mechanics
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_PY_GLOBALS_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_PY_GLOBALS_PUBLIC_HEADER_FILES})

# Install the library file of engines_mechanics
install(TARGETS engines_py_globals 
  CONFIGURATIONS RELEASE    
  DESTINATION lib
)

install(TARGETS engines_py_globals
  CONFIGURATIONS DEBUG     
  DESTINATION lib
)

# # Install the public header files
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
#         DESTINATION include
#         COMPONENT header_files
#         FILES_MATCHING PATTERN "*.hpp")

# ------------------------------------------------------------------------------
