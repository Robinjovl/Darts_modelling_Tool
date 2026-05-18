# Build solvers python interface -------------------------------------------------------
# Set library files

# First the source files (private headers should also go here)
set(OD_SOLVERS_PYBIND11_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/py_main.cpp
)

# Then the public headers (if any)
set(OD_SOLVERS_PYBIND11_PUBLIC_HEADER_FILES
)

# Define the solvers Python extension.
# The compiled module is named `_solvers` and is installed into the darts/solvers
# package directory, whose __init__.py re-exports it. This frees the name
# `darts.solvers` for the Python package (LinearSolverSpec classes, enums).
pybind11_add_module(_solvers
  ${OD_SOLVERS_PYBIND11_SRC_FILES}
  ${OD_SOLVERS_PYBIND11_PUBLIC_HEADER_FILES}
)

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(_solvers
    PUBLIC
      OPENDARTS_LINEAR_SOLVERS
      PYBIND11_ENABLED
    )
else()
  target_compile_definitions(_solvers
    PUBLIC
      PYBIND11_ENABLED
    )
endif()

target_link_libraries(_solvers
  PUBLIC
    opendarts_compiler_flags
    opendarts_solvers
    pybind11::pybind11
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(_solvers
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/solvers/include>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for solvers
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_SOLVERS_PYBIND11_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_SOLVERS_PYBIND11_PUBLIC_HEADER_FILES})

# Install the compiled module into the darts/solvers package directory.
install(TARGETS _solvers
  CONFIGURATIONS RELEASE RELWITHDEBINFO DEBUG ""
  DESTINATION ./solvers
)

# ------------------------------------------------------------------------------
