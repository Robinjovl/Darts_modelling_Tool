# Build python interface -------------------------------------------------------
# Set library files

# First the source files (private headers should also go here)
set(OD_ENGINES_PYBIND11_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/py_engine_base.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_engine_nc_cg_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_engine_nc_nl_cpu.cpp # non-linear  
  ${CMAKE_CURRENT_SOURCE_DIR}/py_engine_nce_g_cpu.cpp # geothermal
  ${CMAKE_CURRENT_SOURCE_DIR}/py_engine_super_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_engine_super_elastic_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_engine_elasticity_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_engine_pm_cpu.cpp  # poromechanics  
  ${CMAKE_CURRENT_SOURCE_DIR}/py_engine_super_mp_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_evaluator_iface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_globals.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_ms_well.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_pz_cap_gra.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_pze_gra.cpp  # geothermal
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_super.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_rates.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_well_controls.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_contact.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_mesh_conn.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_mech_operators.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_pm_discretizer.cpp
)

# Then the public headers
set(OD_ENGINES_PYBIND11_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/py_globals.h
  ${CMAKE_CURRENT_SOURCE_DIR}/py_globals_interpolation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/py_evaluator_iface.h
  ${CMAKE_CURRENT_SOURCE_DIR}/py_interpolator_exposer.hpp
)

# Define the engines library
pybind11_add_module(engines
  ${OD_ENGINES_PYBIND11_SRC_FILES}
  ${OD_ENGINES_PYBIND11_PUBLIC_HEADER_FILES}
)

if (CUDA)
  set(CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/py_engine_nc_cg_gpu.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/py_engine_super_gpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_gpu_tools.cpp
  )

  set_source_files_properties(
    ${CUDA_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_pz_cap_gra.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_pze_gra.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_super.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_operator_set_interpolator_rates.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/py_engine_nce_g_cpu.cpp

    PROPERTIES LANGUAGE CUDA)
  
  target_sources(engines PRIVATE
    ${CUDA_SOURCES}
  )

  # The following property may speed up compilation, at the cost of some performance
  # set_target_properties(engines PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with 
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(engines 
    PUBLIC 
      OPENDARTS_LINEAR_SOLVERS
      PYBIND11_ENABLED
    )
else()
  target_compile_definitions(engines 
    PUBLIC
      PYBIND11_ENABLED
    )
endif()

target_link_libraries(engines
  PUBLIC
    opendarts_compiler_flags
    engines_global
    engines_wells
    engines_build_info
    engines_engines
    engines_mechanics
    engines_interpolator
    engines_conn_mesh
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(engines
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for engines
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_PYBIND11_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_PYBIND11_PUBLIC_HEADER_FILES})

# Install the library file of engines
install(TARGETS engines 
  CONFIGURATIONS RELEASE RELWITHDEBINFO DEBUG ""
  DESTINATION ./
)

# # Install the public header files
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
#         DESTINATION include
#         COMPONENT header_files
#         FILES_MATCHING PATTERN "*.hpp")

# ------------------------------------------------------------------------------
