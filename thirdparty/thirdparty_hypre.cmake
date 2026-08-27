# Hypre ------------------------------------------------------------------------
# Imports Hypre library so that they can be used in the project.
# ------------------------------------------------------------------------------

# Initialize reporting ---------------------------------------------------------
message(CHECK_START "   Importing Hypre")

if (NOT DEFINED HYPRE_DIR)
  message(STATUS "      Setting default HYPRE_DIR")
  if(EXISTS "${CMAKE_SOURCE_DIR}/thirdparty/install/lib/cmake/HYPRE")
    set(HYPRE_DIR "${CMAKE_SOURCE_DIR}/thirdparty/install/lib/cmake/HYPRE")
  elseif(EXISTS "${CMAKE_SOURCE_DIR}/thirdparty/install/lib64/cmake/HYPRE")
    set(HYPRE_DIR "${CMAKE_SOURCE_DIR}/thirdparty/install/lib64/cmake/HYPRE")
  else()
    message(FATAL_ERROR "HYPRE directory does not exist in lib or lib64.")
  endif()
endif(NOT DEFINED HYPRE_DIR)

# find_package requires absolute paths to work, make sure the path is absolute
message(STATUS "      Converting Hypre search path to absolute path")
message(STATUS "         Input path: ${HYPRE_DIR}")
file(REAL_PATH "${HYPRE_DIR}" HYPRE_DIR BASE_DIRECTORY "${CMAKE_BINARY_DIR}")
message(STATUS "         Absolute path: ${HYPRE_DIR}")

# A HYPRE built with HYPRE_ENABLE_OPENMP=ON (the HYPRE_OPENMP=1 build option)
# exports a link dependency on the OpenMP::OpenMP_C imported target, so that
# target must exist before HYPRE is imported. Harmless for a sequential HYPRE
# (the target is simply unused). Not REQUIRED, so a sequential build on a
# toolchain without OpenMP still configures.
find_package(OpenMP)

# Find Hypre
find_package(HYPRE REQUIRED CONFIG)
if (TARGET HYPRE::HYPRE)
  message(STATUS "      Found Hypre: TRUE")
else()
  message(FATAL_ERROR "      Found Hypre: FALSE")
endif (TARGET HYPRE::HYPRE)

# Some user feedback info
get_target_property(HYPRE_INCLUDE_DIRS HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)
set(HYPRE_LIBRARY_PATH "")
string(TOUPPER "${CMAKE_BUILD_TYPE}" HYPRE_BUILD_TYPE_UPPER)
if(HYPRE_BUILD_TYPE_UPPER)
  get_target_property(HYPRE_LIBRARY_PATH HYPRE::HYPRE "IMPORTED_LOCATION_${HYPRE_BUILD_TYPE_UPPER}")
endif()
if(NOT HYPRE_LIBRARY_PATH)
  get_target_property(HYPRE_LIBRARY_PATH HYPRE::HYPRE IMPORTED_LOCATION)
endif()
if(NOT HYPRE_LIBRARY_PATH)
  get_target_property(HYPRE_LIBRARY_PATH HYPRE::HYPRE IMPORTED_LOCATION_RELEASE)
endif()
message(STATUS "      Include directories: ${HYPRE_INCLUDE_DIRS}")
message(STATUS "      Library path       : ${HYPRE_LIBRARY_PATH}")

# Finalize reporting and check if all libraries have been added ----------------
message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------
