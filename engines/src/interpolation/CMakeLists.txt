# Build opendarts_engines_interpolators ------------------------------------------------

# Set library files

# First the source files (private headers should also go here)
set(OD_ENGINES_INTERPOLATOR_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/interpolator_base.cpp
)

# Then the public headers
set(OD_ENGINES_INTERPOLATOR_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/interpolator_base.hpp
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
  ${CMAKE_CURRENT_SOURCE_DIR}/evaluator_iface.h    
)




# Define the engines_interpolator library
# This is always a static library since we wish to use it internally only
add_library(engines_interpolator OBJECT "")

if (CUDA)
  set(CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/evaluator_iface.cpp 
  )
  set_source_files_properties(
    ${CUDA_SOURCES}
    PROPERTIES LANGUAGE CUDA)
  
  target_sources(engines_interpolator PRIVATE
    ${CUDA_SOURCES}
  )
endif()

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with 
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(engines_interpolator PUBLIC OPENDARTS_LINEAR_SOLVERS)
endif()

target_sources(engines_interpolator
  PRIVATE
    ${OD_ENGINES_INTERPOLATOR_SRC_FILES}
    ${OD_ENGINES_INTERPOLATOR_PUBLIC_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
)

target_link_libraries(engines_interpolator
  PUBLIC
    opendarts_compiler_flags
    linear_solvers
    engines_global
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(engines_interpolator
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for engines_interpolator
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_INTERPOLATOR_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_INTERPOLATOR_PUBLIC_HEADER_FILES})

# Install the library file of engines_interpolator
install(TARGETS engines_interpolator 
  CONFIGURATIONS RELEASE    
  DESTINATION lib
)

install(TARGETS engines_interpolator
  CONFIGURATIONS DEBUG     
  DESTINATION lib
)

# # Install the public header files
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
#         DESTINATION include
#         COMPONENT header_files
#         FILES_MATCHING PATTERN "*.hpp")

# ------------------------------------------------------------------------------
