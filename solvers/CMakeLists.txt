# Give the user some feedback on what is going on
message(STATUS "Building openDARTS solvers")

# Get hostname and username ----------------------------------------------------
cmake_host_system_information(RESULT opendarts_linear_solvers_BUILD_HOSTNAME QUERY HOSTNAME)
if(WIN32)
  set(opendarts_linear_solvers_BUILD_USERNAME $ENV{USERNAME})
else()
  set(opendarts_linear_solvers_BUILD_USERNAME $ENV{USER})
endif()
# ------------------------------------------------------------------------------


# Setup testing ----------------------------------------------------------------

# Report Enable or disable testing via option flag
if(ENABLE_TESTING)
  enable_testing()
endif()
# ------------------------------------------------------------------------------

# Include solvers sources (config, auxiliary, linear_solvers - all unified) ----
add_subdirectory(src)
# ------------------------------------------------------------------------------

# Create shared library that aggregates all solver components --------------------
# This is the SINGLE shared library for all solvers functionality
# Having everything in one shared library avoids ODR violations and
# ensures HYPRE's static state is consistent across all solver code.
add_library(opendarts_solvers SHARED "")

# Link the unified linear_solvers library (which includes MGR)
target_link_libraries(opendarts_solvers
  PUBLIC
    opendarts_compiler_flags
    config
    auxiliary
    linear_solvers
)

# Include directories
target_include_directories(opendarts_solvers
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Alias for easier linking
add_library(opendarts::solvers ALIAS opendarts_solvers)

# Install the shared solvers library next to Python modules
install(TARGETS opendarts_solvers
  CONFIGURATIONS RELEASE RELWITHDEBINFO DEBUG ""
  DESTINATION ./
)

message(STATUS "Shared solvers library configured (unified)")

# Include Python bindings ------------------------------------------------------
add_subdirectory(src/pybind11)
# ------------------------------------------------------------------------------

message(STATUS "Building openDARTS solvers - done!")
