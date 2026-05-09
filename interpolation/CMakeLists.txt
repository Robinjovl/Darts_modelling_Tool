# Build openDARTS interpolation library -----------------------------------------
message(STATUS "Building openDARTS interpolation library")

# Core interpolation library (OBJECT)
add_subdirectory(src)

# Python bindings (pybind11 module)
add_subdirectory(pybind11)

message(STATUS "Building openDARTS interpolation library - done!")
