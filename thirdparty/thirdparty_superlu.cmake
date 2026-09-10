# SuperLU ----------------------------------------------------------------------
# Imports SuperLU so that it can be used in the project.
#
# SuperLU is consumed as a pinned git submodule (thirdparty/superlu, upstream
# xiaoyeli/superlu) that the build scripts compile with SuperLU's own CMake and
# install into thirdparty/install -- exactly like HYPRE. Here we simply locate
# that installed package and expose its CONFIG target superlu::superlu (which
# transitively carries the bundled reference CBLAS `blas` target, so no system
# BLAS is required and the build stays self-contained).
# ------------------------------------------------------------------------------

# Initialize reporting ---------------------------------------------------------
message(CHECK_START "   Importing SuperLU")

# Locate the installed superlu CMake package (mirror of thirdparty_hypre.cmake).
# GNUInstallDirs may place the package config under lib or lib64.
if (NOT DEFINED superlu_DIR)
  message(STATUS "      Setting default superlu_DIR")
  if(EXISTS "${CMAKE_SOURCE_DIR}/thirdparty/install/lib/cmake/superlu")
    set(superlu_DIR "${CMAKE_SOURCE_DIR}/thirdparty/install/lib/cmake/superlu")
  elseif(EXISTS "${CMAKE_SOURCE_DIR}/thirdparty/install/lib64/cmake/superlu")
    set(superlu_DIR "${CMAKE_SOURCE_DIR}/thirdparty/install/lib64/cmake/superlu")
  else()
    message(FATAL_ERROR "SuperLU package not found under thirdparty/install/lib(64)/cmake/superlu. Build thirdparty first (helper_scripts/build_darts_cmake.sh, with -c to force a fresh thirdparty build).")
  endif()
endif(NOT DEFINED superlu_DIR)

# find_package requires absolute paths to work, make sure the path is absolute
file(REAL_PATH "${superlu_DIR}" superlu_DIR BASE_DIRECTORY "${CMAKE_BINARY_DIR}")
message(STATUS "      SuperLU search path: ${superlu_DIR}")

find_package(superlu REQUIRED CONFIG)
if (TARGET superlu::superlu)
  message(STATUS "      Found SuperLU: TRUE")
else()
  message(FATAL_ERROR "      Found SuperLU: FALSE")
endif (TARGET superlu::superlu)

message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------
