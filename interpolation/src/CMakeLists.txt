# Build darts_interpolation core library ----------------------------------------

set(INTERPOLATION_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Source files
set(INTERPOLATION_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/interpolator_base.cpp
)

# Public headers
set(INTERPOLATION_PUBLIC_HEADER_FILES
  ${INTERPOLATION_INCLUDE_DIR}/timer_node.h
  ${INTERPOLATION_INCLUDE_DIR}/interpolation_config.h
  ${INTERPOLATION_INCLUDE_DIR}/recursive_exposers.h
  ${INTERPOLATION_INCLUDE_DIR}/evaluator_iface.h
  ${INTERPOLATION_INCLUDE_DIR}/interpolator_base.hpp
  ${INTERPOLATION_INCLUDE_DIR}/matrix.h
  ${INTERPOLATION_INCLUDE_DIR}/linear_adaptive_cpu_interpolator.hpp
  ${INTERPOLATION_INCLUDE_DIR}/linear_adaptive_cpu_interpolator.tpp
  ${INTERPOLATION_INCLUDE_DIR}/linear_cpu_interpolator_base.hpp
  ${INTERPOLATION_INCLUDE_DIR}/linear_cpu_interpolator_base.tpp
  ${INTERPOLATION_INCLUDE_DIR}/linear_static_cpu_interpolator.hpp
  ${INTERPOLATION_INCLUDE_DIR}/linear_static_cpu_interpolator.tpp
  ${INTERPOLATION_INCLUDE_DIR}/multilinear_adaptive_cpu_interpolator.hpp
  ${INTERPOLATION_INCLUDE_DIR}/multilinear_adaptive_cpu_interpolator.tpp
  ${INTERPOLATION_INCLUDE_DIR}/multilinear_interpolator_base.hpp
  ${INTERPOLATION_INCLUDE_DIR}/multilinear_interpolator_base.tpp
  ${INTERPOLATION_INCLUDE_DIR}/multilinear_interpolator_common.h
  ${INTERPOLATION_INCLUDE_DIR}/multilinear_static_cpu_interpolator.hpp
  ${INTERPOLATION_INCLUDE_DIR}/multilinear_static_cpu_interpolator.tpp
)

# Define the library as OBJECT (same pattern as the old engines_interpolator)
add_library(darts_interpolation OBJECT "")

if(NOT ENABLE_BOS_SOLVERS)
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
    ${INTERPOLATION_INCLUDE_DIR}/gpu_hashmap_async.h
    ${INTERPOLATION_INCLUDE_DIR}/gpu_hashmap_async.tpp
    ${INTERPOLATION_INCLUDE_DIR}/gpu_tools.h
    ${INTERPOLATION_INCLUDE_DIR}/multilinear_gpu_interpolator_base.hpp
    ${INTERPOLATION_INCLUDE_DIR}/multilinear_gpu_interpolator_base.tpp
    ${INTERPOLATION_INCLUDE_DIR}/multilinear_adaptive_gpu_interpolator.hpp
    ${INTERPOLATION_INCLUDE_DIR}/multilinear_adaptive_gpu_interpolator.tpp
    ${INTERPOLATION_INCLUDE_DIR}/multilinear_static_gpu_interpolator.hpp
    ${INTERPOLATION_INCLUDE_DIR}/multilinear_static_gpu_interpolator.tpp
  )

  target_link_libraries(darts_interpolation
    PRIVATE
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
    $<BUILD_INTERFACE:${INTERPOLATION_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${INTERPOLATION_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${INTERPOLATION_PUBLIC_HEADER_FILES})


# ------------------------------------------------------------------------------
