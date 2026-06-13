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
# The compiled module is named `solvers` and is installed into the darts/solvers
# package directory (as darts/solvers/solvers*.so). The package __init__.py
# re-exports it, so `import darts.solvers` exposes the compiled API while the
# package also hosts the Python LinearSolverSpec classes and enums.
pybind11_add_module(solvers
  ${OD_SOLVERS_PYBIND11_SRC_FILES}
  ${OD_SOLVERS_PYBIND11_PUBLIC_HEADER_FILES}
)

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT ENABLE_BOS_SOLVERS)
  target_compile_definitions(solvers
    PUBLIC
      OPENDARTS_LINEAR_SOLVERS
      PYBIND11_ENABLED
    )
else()
  target_compile_definitions(solvers
    PUBLIC
      PYBIND11_ENABLED
    )
endif()

target_link_libraries(solvers
  PUBLIC
    opendarts_compiler_flags
    opendarts_solvers
    pybind11::pybind11
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(solvers
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/solvers/include>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for solvers
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_SOLVERS_PYBIND11_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_SOLVERS_PYBIND11_PUBLIC_HEADER_FILES})

# Install the compiled module into the darts/solvers package directory.
install(TARGETS solvers
  CONFIGURATIONS RELEASE RELWITHDEBINFO DEBUG ""
  DESTINATION ./solvers
)

# ------------------------------------------------------------------------------
