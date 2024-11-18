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

# Include config library -------------------------------------------------------
add_subdirectory(config)
# ------------------------------------------------------------------------------


# Include auxiliary library ----------------------------------------------------
add_subdirectory(auxiliary)
# ------------------------------------------------------------------------------


# Include linear_solvers library -----------------------------------------------
add_subdirectory(linear_solvers)
# ------------------------------------------------------------------------------

message(STATUS "Building openDARTS solvers - done!")  