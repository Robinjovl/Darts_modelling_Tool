# Pybind11 ---------------------------------------------------------------------
# Imports pybind11 libraries so that they can be used in the project.
# ------------------------------------------------------------------------------

# Initialize reporting ---------------------------------------------------------
# Reports we started looking for Eigen library and initializes the flag that
# checks if all have been found 
message(CHECK_START "   Importing pybind11")
# ------------------------------------------------------------------------------

# Adds pybind11 ----------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

if(DEFINED pybind11_DIR)
  set(pybind11_BINARY_DIR "${pybind11_DIR}/build/pybind11" )
else()
  set(pybind11_DIR "${CMAKE_CURRENT_LIST_DIR}/pybind11")
  set(pybind11_BINARY_DIR "${CMAKE_CURRENT_LIST_DIR}/build/pybind11" )
endif(DEFINED pybind11_DIR)

# find_package requires absolute paths to work, make sure the path is absolute
message(STATUS "      Converting pybind11 search path to absolute path")
message(STATUS "         Input path: ${pybind11_DIR}")
file(REAL_PATH "${pybind11_DIR}" pybind11_DIR BASE_DIRECTORY "${CMAKE_BINARY_DIR}")
file(REAL_PATH "${pybind11_BINARY_DIR}" pybind11_BINARY_DIR BASE_DIRECTORY "${CMAKE_BINARY_DIR}")
message(STATUS "         Absolute path: ${pybind11_DIR}")
message(STATUS "         Absolute path BINARY_DIR: ${pybind11_BINARY_DIR}")

add_subdirectory(${pybind11_DIR} ${pybind11_BINARY_DIR})

# Some user feedback info
message(STATUS "      Include directories: ${pybind11_INCLUDE_DIRS}")

# Finalize reporting -----------------------------------------------------------
message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------