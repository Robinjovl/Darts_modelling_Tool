# Build darts_interpolation Python module ----------------------------------------

# Source files
set(INTERPOLATION_PYBIND11_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/py_evaluator_iface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_all.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_d1_d2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_d3_d4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_d5_d6.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_d7_d8.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_d9_d10.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_main_interpolation.cpp
)

# Public headers
set(INTERPOLATION_PYBIND11_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/py_globals_interpolation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/py_evaluator_iface.h
  ${CMAKE_CURRENT_SOURCE_DIR}/py_interpolator_exposer.hpp
)

# Define the pybind11 module
pybind11_add_module(darts_interpolation_py
  ${INTERPOLATION_PYBIND11_SRC_FILES}
  ${INTERPOLATION_PYBIND11_PUBLIC_HEADER_FILES}
)

if (CUDA)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_all.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_d1_d2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_d3_d4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_d5_d6.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_d7_d8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_d9_d10.cpp
    PROPERTIES LANGUAGE CUDA)
endif()

if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(darts_interpolation_py
    PUBLIC
      OPENDARTS_LINEAR_SOLVERS
      PYBIND11_ENABLED
  )
else()
  target_compile_definitions(darts_interpolation_py
    PUBLIC
      PYBIND11_ENABLED
  )
endif()

target_link_libraries(darts_interpolation_py
  PUBLIC
    opendarts_compiler_flags
    darts_interpolation
)

target_include_directories(darts_interpolation_py
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${INTERPOLATION_PYBIND11_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${INTERPOLATION_PYBIND11_PUBLIC_HEADER_FILES})

# Install the module
install(TARGETS darts_interpolation_py
  CONFIGURATIONS RELEASE RELWITHDEBINFO DEBUG ""
  DESTINATION ./
)

# ------------------------------------------------------------------------------
