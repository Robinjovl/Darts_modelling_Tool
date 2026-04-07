# Build darts_interpolation core library ----------------------------------------

# Source files
set(INTERPOLATION_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/interpolator_base.cpp
)

# Public headers
set(INTERPOLATION_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/interpolation_config.h
  ${CMAKE_CURRENT_SOURCE_DIR}/recursive_exposers.h
  ${CMAKE_CURRENT_SOURCE_DIR}/evaluator_iface.h
  ${CMAKE_CURRENT_SOURCE_DIR}/interpolator_base.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matrix.h
  ${CMAKE_CURRENT_SOURCE_DIR}/linear_adaptive_cpu_interpolator.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linear_adaptive_cpu_interpolator.tpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linear_cpu_interpolator_base.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linear_cpu_interpolator_base.tpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linear_static_cpu_interpolator.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linear_static_cpu_interpolator.tpp
  ${CMAKE_CURRENT_SOURCE_DIR}/multilinear_adaptive_cpu_interpolator.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/multilinear_adaptive_cpu_interpolator.tpp
  ${CMAKE_CURRENT_SOURCE_DIR}/multilinear_interpolator_base.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/multilinear_interpolator_base.tpp
  ${CMAKE_CURRENT_SOURCE_DIR}/multilinear_interpolator_common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/multilinear_static_cpu_interpolator.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/multilinear_static_cpu_interpolator.tpp
)

# Define the library as OBJECT (same pattern as the old engines_interpolator)
add_library(darts_interpolation OBJECT "")

if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(darts_interpolation PUBLIC OPENDARTS_LINEAR_SOLVERS)
endif()

if (CUDA)
  set(CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/evaluator_iface.cpp
  )
  set_source_files_properties(
    ${CUDA_SOURCES}
    PROPERTIES LANGUAGE CUDA)

  target_sources(darts_interpolation PRIVATE
    ${CUDA_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_hashmap_async.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_hashmap_async.tpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multilinear_gpu_interpolator_base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multilinear_gpu_interpolator_base.tpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multilinear_adaptive_gpu_interpolator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multilinear_adaptive_gpu_interpolator.tpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multilinear_static_gpu_interpolator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multilinear_static_gpu_interpolator.tpp
  )

  target_link_libraries(darts_interpolation
    PUBLIC
      CUDA::cudart
  )
endif()

target_sources(darts_interpolation
  PRIVATE
    ${INTERPOLATION_SRC_FILES}
    ${INTERPOLATION_PUBLIC_HEADER_FILES}
)

target_link_libraries(darts_interpolation
  PUBLIC
    opendarts_compiler_flags
    linear_solvers
    pybind11::pybind11
)

target_include_directories(darts_interpolation
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${INTERPOLATION_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${INTERPOLATION_PUBLIC_HEADER_FILES})


# ------------------------------------------------------------------------------
