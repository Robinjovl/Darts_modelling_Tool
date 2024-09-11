# Build discretizer python interface -------------------------------------------
# Set library files

# First the source files (private headers should also go here)
set(DISCRETIZER_PYBIND11_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/py_approximation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_mech_discretizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_discretizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_elem.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_linalg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_mesh.cpp
)

# Then the public headers
set(DISCRETIZER_PYBIND11_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/py_mech_discretizer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/py_discretizer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/py_elem.h
  ${CMAKE_CURRENT_SOURCE_DIR}/py_global.h
  ${CMAKE_CURRENT_SOURCE_DIR}/py_linalg.h
  ${CMAKE_CURRENT_SOURCE_DIR}/py_mesh.h
)

# Define the discretizer library
pybind11_add_module(discretizer
  ${DISCRETIZER_PYBIND11_SRC_FILES}
  ${DISCRETIZER_PYBIND11_PUBLIC_HEADER_FILES}
)

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with 
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(discretizer 
  PUBLIC 
    OPENDARTS_LINEAR_SOLVERS
    PYBIND11_ENABLED
  )
else()
  target_compile_definitions(discretizer 
  PUBLIC 
    PYBIND11_ENABLED
  )
endif()



target_link_libraries(discretizer
  PUBLIC
    opendarts_compiler_flags
    opendarts_discretizer
    discretizer_build_info
    discretizer_mesh
    discretizer_linalg
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(discretizer
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for engines_mechanics
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${DISCRETIZER_PYBIND11_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${DISCRETIZER_PYBIND11_PUBLIC_HEADER_FILES})

# Install the library file of engines_mechanics
install(TARGETS discretizer 
  CONFIGURATIONS RELEASE RELWITHDEBINFO DEBUG ""
  DESTINATION ./
)

# ------------------------------------------------------------------------------