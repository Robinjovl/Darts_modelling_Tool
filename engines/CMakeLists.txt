# Give the user some feedback on what is going on
message(STATUS "Building openDARTS engines")  

# Get hostname and username 
cmake_host_system_information(RESULT opendarts_engines_BUILD_HOSTNAME QUERY HOSTNAME)
if(WIN32)
  set(opendarts_engines_BUILD_USERNAME $ENV{USERNAME})
else()
  set(opendarts_engines_BUILD_USERNAME $ENV{USER})
endif()

# opendarts-engines ------------------------------------------------------------
add_subdirectory(src)
# ------------------------------------------------------------------------------

message(STATUS "Building openDARTS engines - done!") 