# Eigen ------------------------------------------------------------------------
# Imports Eigen library so that they can be used in the project.
# ------------------------------------------------------------------------------

# Initialize reporting ---------------------------------------------------------
# Reports we started looking for Eigen library and initializes the flag that
# checks if all have been found 
message(CHECK_START "   Importing Eigen3")
unset(thirdparty_missing_components)
# ------------------------------------------------------------------------------

# Adds Eigen -------------------------------------------------------------------
if(NOT DEFINED Eigen3_DIR)
  set(Eigen3_DIR "${CMAKE_SOURCE_DIR}/thirdparty/install/share/eigen3/cmake/")
endif(NOT DEFINED Eigen3_DIR)

# find_package requires absolute paths to work, make sure the path is absolute
message(STATUS "      Converting Eigen3 search path to absolute path")
message(STATUS "         Input path: ${Eigen3_DIR}")
file(REAL_PATH "${Eigen3_DIR}" Eigen3_DIR BASE_DIRECTORY "${CMAKE_BINARY_DIR}")
message(STATUS "         Absolute path: ${Eigen3_DIR}")

# Find Eigen3
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
if (TARGET Eigen3::Eigen)
  message(STATUS "      Found Eigen3: TRUE")
else()
  message(FATAL_ERROR "      Found Eigen3: FALSE")
endif (TARGET Eigen3::Eigen)
# find_package(PkgConfig)
# pkg_search_module(Eigen3 REQUIRED eigen3)

# Some user feedback info
get_target_property(EIGEN3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "      Include directories: ${EIGEN3_INCLUDE_DIRS}")

# Finalize reporting -----------------------------------------------------------
message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------