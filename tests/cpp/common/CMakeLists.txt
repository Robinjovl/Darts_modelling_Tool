# Build libtest_common ---------------------------------------------------------

# Set library files

# First the source files (private headers should also go here)
set(OD_LINEAR_SOLVERS_COMMON_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/test_common.cpp
)

# Then the public headers
set(OD_LINEAR_SOLVERS_COMMON_PUBLIC_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/test_common.hpp
)

# Define the libtest_common library
# This is always a static library since we wish to use it internally only
add_library(linear_solvers_test_common STATIC "")

target_sources(linear_solvers_test_common
  PRIVATE
    ${OD_LINEAR_SOLVERS_COMMON_SRC_FILES}
    ${OD_LINEAR_SOLVERS_COMMON_PUBLIC_HEADER_FILES}  # if added as PUBLIC appears in all targets that link to it
)

target_link_libraries(linear_solvers_test_common
    opendarts_compiler_flags
    linear_solvers
)

# Define the include directories required internally (BUILD_INTERFACE)
# and the ones to propagate when linking to this library (INSTALL_INTERFACE)
target_include_directories(linear_solvers_test_common
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# Setup IDE directory structure for opendarts_linear_solvers_test_common
set_property(TARGET linear_solvers_test_common PROPERTY FOLDER "tests//common")
source_group(src FILES ${OD_LINEAR_SOLVERS_COMMON_SRC_FILES})
source_group(src FILES ${OD_LINEAR_SOLVERS_COMMON_PUBLIC_HEADER_FILES})

# ------------------------------------------------------------------------------
