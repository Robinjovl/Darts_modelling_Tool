# Build darts_interpolation Python module ----------------------------------------

# Source files
set(INTERPOLATION_PYBIND11_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/py_evaluator_iface.cpp
  # Split interpolator_all into 5 parts by N_DIMS range for parallel compilation
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_all.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_all_part1.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_all_part2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_all_part3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_all_part4.cpp
  # Split interpolator_super into 4 parts for parallel compilation
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_super.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_super_part1.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_super_part2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_super_part3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_super_part4.cpp
  # Split interpolator_rates into 3 parts for parallel compilation
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_rates.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_rates_part1.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_rates_part2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_rates_part3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_pz_cap_gra.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_pze_gra.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_main_interpolation.cpp
)

# Public headers
set(INTERPOLATION_PYBIND11_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/py_globals_interpolation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/py_evaluator_iface.h
  ${CMAKE_CURRENT_SOURCE_DIR}/py_interpolator_exposer.hpp
)

# Define the pybind11 module
pybind11_add_module(interpolators
  ${INTERPOLATION_PYBIND11_SRC_FILES}
  ${INTERPOLATION_PYBIND11_PUBLIC_HEADER_FILES}
)

if (CUDA)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_all.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_all_part1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_all_part2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_all_part3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_all_part4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_super.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_super_part1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_super_part2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_super_part3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_super_part4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_rates.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_rates_part1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_rates_part2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_rates_part3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_pz_cap_gra.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_pze_gra.cpp
    PROPERTIES LANGUAGE CUDA)
endif()

if(NOT ENABLE_BOS_SOLVERS)
  target_compile_definitions(interpolators
    PUBLIC
      OPENDARTS_LINEAR_SOLVERS
      PYBIND11_ENABLED
  )
else()
  target_compile_definitions(interpolators
    PUBLIC
      PYBIND11_ENABLED
  )
endif()

target_link_libraries(interpolators
  PUBLIC
    opendarts_compiler_flags
    darts_interpolation
)

target_include_directories(interpolators
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${INTERPOLATION_PYBIND11_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${INTERPOLATION_PYBIND11_PUBLIC_HEADER_FILES})

# Install the module
install(TARGETS interpolators
  CONFIGURATIONS RELEASE RELWITHDEBINFO DEBUG ""
  DESTINATION ./
)

# ------------------------------------------------------------------------------
