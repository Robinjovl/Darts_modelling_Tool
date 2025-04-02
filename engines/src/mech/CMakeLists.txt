# Build opendarts_engines_mechanics --------------------------------------------
# Set library files

# First the source files (private headers should also go here)
set(OD_ENGINES_MECHANICS_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_elasticity_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pm_discretizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mech_operators.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_pm_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/contact.cpp
)

# Then the public headers
set(OD_ENGINES_MECHANICS_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_elasticity_cpu.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pm_discretizer.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mech_operators.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_pm_cpu.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/contact.h  
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_super_elastic_cpu.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_super_elastic_cpu.tpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matrix.h
)

# Define the engines_mechanics library
# This is always a static library since we wish to use it internally only
add_library(engines_mechanics OBJECT "")

# We need to specify OPENDARTS_LINEAR_SOLVERS so that the compile is with 
# the opendarts linear solvers, otherwise it uses the old linear solvers
if(NOT DEFINED BOS_SOLVERS_DIR)
  target_compile_definitions(engines_mechanics PUBLIC OPENDARTS_LINEAR_SOLVERS)
endif()

target_sources(engines_mechanics
  PRIVATE
    ${OD_ENGINES_MECHANICS_SRC_FILES}
    ${OD_ENGINES_MECHANICS_PUBLIC_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
	${HYPRE_INCLUDE_DIRS}
)

target_link_libraries(engines_mechanics
  PUBLIC
    opendarts_compiler_flags
    linear_solvers
    engines_global
    engines_interpolator
	opendarts_discretizer
    Eigen3::Eigen
	HYPRE::HYPRE
    pybind11::pybind11
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(engines_mechanics
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Setup IDE directory structure for engines_mechanics
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_MECHANICS_SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${OD_ENGINES_MECHANICS_PUBLIC_HEADER_FILES})

# Install the library file of engines_mechanics
install(TARGETS engines_mechanics 
  CONFIGURATIONS RELEASE    
  DESTINATION lib
)

install(TARGETS engines_mechanics
  CONFIGURATIONS DEBUG     
  DESTINATION lib
)

# # Install the public header files
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
#         DESTINATION include
#         COMPONENT header_files
#         FILES_MATCHING PATTERN "*.hpp")

# ------------------------------------------------------------------------------
