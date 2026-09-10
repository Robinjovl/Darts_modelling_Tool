# Build linear_solvers python interface -------------------------------------------------------
# Set library files

# First the source files (private headers should also go here)
set(OD_SOLVERS_PYBIND11_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/py_main.cpp
)

# Then the public headers (if any)
set(OD_SOLVERS_PYBIND11_PUBLIC_HEADER_FILES
)

# Define the linear_solvers Python extension.
# The compiled module is named `linear_solvers` and is installed into the
# darts/linear_solvers package directory (as darts/linear_solvers/linear_solvers*.so).
# The package __init__.py re-exports it, so `import darts.linear_solvers` exposes the
# compiled API while the package also hosts the Python LinearSolverSpec classes and enums.
# The CMake target is `linear_solvers_python` because the plain name `linear_solvers`
# is already taken by the static C++ library; OUTPUT_NAME keeps the module file name.
pybind11_add_module(linear_solvers_python
  ${OD_SOLVERS_PYBIND11_SRC_FILES}
  ${OD_SOLVERS_PYBIND11_PUBLIC_HEADER_FILES}
)
set_target_properties(linear_solvers_python PROPERTIES OUTPUT_NAME linear_solvers)

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT ENABLE_BOS_SOLVERS)
  target_compile_definitions(linear_solvers_python
    PUBLIC
      OPENDARTS_LINEAR_SOLVERS
      PYBIND11_ENABLED
    )
else()
  target_compile_definitions(linear_solvers_python
    PUBLIC
      PYBIND11_ENABLED
    )
endif()

target_link_libraries(linear_solvers_python
  PUBLIC
    opendarts_compiler_flags
    opendarts_linear_solvers
    pybind11::pybind11
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(linear_solvers_python
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/linear_solvers/include>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for solvers
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_SOLVERS_PYBIND11_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_SOLVERS_PYBIND11_PUBLIC_HEADER_FILES})

# Install the compiled module into the darts/linear_solvers package directory.
install(TARGETS linear_solvers_python
  CONFIGURATIONS RELEASE RELWITHDEBINFO DEBUG ""
  DESTINATION ./linear_solvers
)

# ------------------------------------------------------------------------------
