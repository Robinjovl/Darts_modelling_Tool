# Thirdparty libraries ---------------------------------------------------------
# Imports all thirdparty libraries so that they can be used in the 
# project.
# ------------------------------------------------------------------------------


# Initialize reporting ---------------------------------------------------------
# Reports we started looking for thirdparty libraries and initializes the flag 
# that checks if all have been found 
message(CHECK_START "Importing thirdparty libraries")
unset(thirdparty_missing_components)
# ------------------------------------------------------------------------------


# Adds SuperLU -----------------------------------------------------------------
message(CHECK_START "   Importing linear_solvers")

add_library(linear_solvers STATIC IMPORTED GLOBAL)
# When setting the library files to import, we need to do it in different ways for
# Windows and linux/macOS
if(import_externals_as_msvc)
  # For Windows we need to set the .lib file, which has a different name from the .a file
  
  # Check if the required files and paths exist
  set(linear_solvers_library_path "${CMAKE_CURRENT_SOURCE_DIR}/opendarts_linear_solvers/lib/darts_linear_solvers.lib")  # set the location of linear_solvers library (full path)
  set(linear_solvers_headers_path "${CMAKE_CURRENT_SOURCE_DIR}/opendarts_linear_solvers/include")  # set the location of linear_solvers header files
  set(linear_solvers_header_file_to_check "openDARTS/linear_solvers/csr_matrix.hpp")  # check for this header file inside the header files to confirm it exists

  if(NOT EXISTS ${linear_solvers_library_path})
    message(FATAL_ERROR "   linear_solvers library not found: ${linear_solvers_library_path}")
  endif()
  
  if(NOT EXISTS "${linear_solvers_headers_path}/${linear_solvers_header_file_to_check}")
    message(FATAL_ERROR "   linear_solvers headers not found in: ${linear_solvers_headers_path}")
  endif()
  
  set_target_properties(linear_solvers
    PROPERTIES
      IMPORTED_LOCATION ${linear_solvers_library_path}
      INTERFACE_INCLUDE_DIRECTORIES ${linear_solvers_headers_path})
else()
  # For Linux and macOS, we provide the .a file
  
  # Check if the required files and paths exist
  set(linear_solvers_library_path "${CMAKE_CURRENT_SOURCE_DIR}/opendarts_linear_solvers/lib/libdarts_linear_solvers.a")  # set the location of linear_solvers library file (full path)
  set(linear_solvers_headers_path "${CMAKE_CURRENT_SOURCE_DIR}/opendarts_linear_solvers/include")  # set the location of linear_solvers header files 
  set(linear_solvers_header_file_to_check "openDARTS/linear_solvers/csr_matrix.hpp")  # check for this header file inside the header files to confirm it exists

  if(NOT EXISTS ${linear_solvers_library_path})
    message(FATAL_ERROR "   linear_solvers library not found: ${linear_solvers_library_path}")
  endif()
  
  if(NOT EXISTS "${linear_solvers_headers_path}/${linear_solvers_header_file_to_check}")
    message(FATAL_ERROR "   linear_solvers headers not found in: ${linear_solvers_headers_path}")
  endif()
  
  set_target_properties(linear_solvers
    PROPERTIES
      IMPORTED_LOCATION ${linear_solvers_library_path}
    INTERFACE_INCLUDE_DIRECTORIES ${linear_solvers_headers_path})
endif()

message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------


# Finalize reporting and check if all libraries have been added ----------------
message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------
