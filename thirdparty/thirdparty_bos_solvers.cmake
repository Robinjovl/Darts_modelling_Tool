# bos_solvers ------------------------------------------------------------------
# Imports  precompiled bos_solvers library so that they can be used in the 
# project.
# ------------------------------------------------------------------------------

# Initialize reporting ---------------------------------------------------------
# Reports we started looking for bos_solvers library and initializes the flag 
# that checks if all have been found 
message(CHECK_START "   Importing bos_solvers")
unset(thirdparty_missing_components)
# ------------------------------------------------------------------------------

# Define bos_solvers library name ----------------------------------------------
# Check the build type:
#   ST: single threaded (Default)
#   MT: multi threaded
#   GPU: GPU support (not available yet)
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(BOS_SOLVERS_SUFFIX "_d")
  message(STATUS "    Fetching bos_solvers with Debug mode")
elseif(${OPENDARTS_CONFIG} STREQUAL "ST")
  set(BOS_SOLVERS_SUFFIX "") 
elseif(${OPENDARTS_CONFIG} STREQUAL "MT")
  set(BOS_SOLVERS_SUFFIX "_mt")
else()
  message(FATAL_ERROR "openDARTS GPU compilation is not supported yet.")
endif()

message(STATUS "    Fetching bos_solvers ${OPENDARTS_CONFIG}")

# Adds bos_solvers -------------------------------------------------------------
add_library(linear_solvers STATIC IMPORTED GLOBAL)

# When setting the library files to import, we need to do it in different ways for
# Windows and linux/macOS
if(import_externals_as_msvc)
  # For windows the path comes with \ instead of / , thus we need to normalize the path.
  cmake_path(CONVERT "${BOS_SOLVERS_DIR}" TO_CMAKE_PATH_LIST BOS_SOLVERS_DIR)
  # For Windows we need to set the .lib file, which has a different name from the .a file
  set(bos_solvers_library_path "${BOS_SOLVERS_DIR}/lib/darts_linear_solvers${BOS_SOLVERS_SUFFIX}.lib")  # set the location of bos_solvers library file (full path)
  set(bos_solvers_headers_path "${BOS_SOLVERS_DIR}/include")  # set the location of bos_solvers header files 
  set(bos_solvers_header_file_to_check "csr_matrix.h")  # check for this header file inside the header files

  if(NOT EXISTS ${bos_solvers_library_path})
    message(FATAL_ERROR "   bos_solvers library not found: ${bos_solvers_library_path}")
  endif()
  
  if(NOT EXISTS "${bos_solvers_headers_path}/${bos_solvers_header_file_to_check}")
    message(FATAL_ERROR "   bos_solver headers not found in: ${bos_solvers_headers_path}")
  endif()
  
  set_target_properties(linear_solvers
    PROPERTIES
      IMPORTED_LOCATION ${bos_solvers_library_path}
      INTERFACE_INCLUDE_DIRECTORIES ${bos_solvers_headers_path})
else()
  # For Linux and macOS, we provide the .a file
  # Check if the required files and paths exist
  set(bos_solvers_library_path "${BOS_SOLVERS_DIR}/lib/libdarts_linear_solvers${BOS_SOLVERS_SUFFIX}.a")  # set the location of bos_solvers library file (full path)
  set(bos_solvers_headers_path "${BOS_SOLVERS_DIR}/include")  # set the location of bos_solvers header files 
  set(bos_solvers_header_file_to_check "csr_matrix.h")  # check for this header file inside the header files  

  if(NOT EXISTS ${bos_solvers_library_path})
    message(FATAL_ERROR "   bos_solvers library not found: ${bos_solvers_library_path}")
  endif()
  
  if(NOT EXISTS "${bos_solvers_headers_path}/${bos_solvers_header_file_to_check}")
    message(FATAL_ERROR "   bos_solvers headers not found in: ${bos_solvers_headers_path}")
  endif()
  
  set_target_properties(linear_solvers
    PROPERTIES
      IMPORTED_LOCATION ${bos_solvers_library_path}
    INTERFACE_INCLUDE_DIRECTORIES ${bos_solvers_headers_path})
endif()

message(STATUS "      bos_solvers DIR: ${bos_solvers_library_path}")
message(STATUS "      bos_solvers Include directories: ${bos_solvers_headers_path}")

# Finalize reporting -----------------------------------------------------------
message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------