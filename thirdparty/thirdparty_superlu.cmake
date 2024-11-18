# SuperLU ----------------------------------------------------------------------
# Imports SuperLU library so that they can be used in the project.
# ------------------------------------------------------------------------------

# Initialize reporting ---------------------------------------------------------
# Reports we started looking for thirdparty libraries and initializes the flag 
# that checks if all have been found 
message(CHECK_START "   Importing SuperLU")
unset(thirdparty_missing_components)
# ------------------------------------------------------------------------------

# Adds SuperLU -----------------------------------------------------------------
add_library(SuperLU STATIC IMPORTED GLOBAL)
# When setting the library files to import, we need to do it in different ways for
# Windows and linux/macOS
if(import_externals_as_msvc)
  # For Windows we need to set the .lib file, which has a different name from the .a file
  
  # Check if the required files and paths exist
  set(SuperLU_library_path "${CMAKE_CURRENT_LIST_DIR}/SuperLU_5.2.1/x64/Release/SuperLU.lib")  # set the location of blas library file (full path)
  set(SuperLU_headers_path "${CMAKE_CURRENT_LIST_DIR}/SuperLU_5.2.1/SRC")  # set the location of blas header files 
  set(SuperLU_header_file_to_check "slu_ddefs.h")  # check for this header file inside the header files  TODO: Need to find a header file that is relevant

  if(NOT EXISTS ${SuperLU_library_path})
    message(FATAL_ERROR "   SuperLU library not found: ${SuperLU_library_path}")
  endif()
  
  if(NOT EXISTS "${SuperLU_headers_path}/${SuperLU_header_file_to_check}")
    message(FATAL_ERROR "   SuperLU headers not found in: ${SuperLU_headers_path}")
  endif()
  
  set(blas_header_file_to_check "slu_ddefs.h")  # check for this header file inside the header files  
  set_target_properties(SuperLU
    PROPERTIES
      IMPORTED_LOCATION ${SuperLU_library_path}
      INTERFACE_INCLUDE_DIRECTORIES ${SuperLU_headers_path})
else()
  # For Linux and macOS, we provide the .a file
  
  # Check if the required files and paths exist
  set(SuperLU_library_path "${CMAKE_CURRENT_LIST_DIR}/SuperLU_5.2.1/libsuperlu_5.1.a")  # set the location of blas library file (full path)
  set(SuperLU_headers_path "${CMAKE_CURRENT_LIST_DIR}/SuperLU_5.2.1/SRC")  # set the location of blas header files 
  set(SuperLU_header_file_to_check "slu_ddefs.h")  # check for this header file inside the header files  TODO: Need to find a header file that is relevant

  if(NOT EXISTS ${SuperLU_library_path})
    message(FATAL_ERROR "   SuperLU library not found: ${SuperLU_library_path}")
  endif()
  
  if(NOT EXISTS "${SuperLU_headers_path}/${SuperLU_header_file_to_check}")
    message(FATAL_ERROR "   SuperLU headers not found in: ${SuperLU_headers_path}")
  endif()
  
  set_target_properties(SuperLU
    PROPERTIES
      IMPORTED_LOCATION ${SuperLU_library_path}
    INTERFACE_INCLUDE_DIRECTORIES ${SuperLU_headers_path})
endif()

message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------

# Adds cBLAS, which is required by SuperLU -------------------------------------
message(CHECK_START "   Importing cblas")

add_library(cblas STATIC IMPORTED GLOBAL)
# When setting the library files to import, we need to do it in different ways for
# Windows and linux/macOS
if(import_externals_as_msvc)
  # For Windows we need to set the .lib file, which has a different name from the .a file
  
  # Check if the required files and paths exist
  set(blas_library_path "${CMAKE_CURRENT_LIST_DIR}/SuperLU_5.2.1/x64/Release/slu_blas.lib")  # set the location of blas library file (full path)
  set(blas_headers_path "${CMAKE_CURRENT_LIST_DIR}/SuperLU_5.2.1/SRC")  # set the location of blas header files 
  # set(blas_header_file_to_check "slu_ddefs.h")  # check for this header file inside the header files  TODO: Need to find a header file that is relevant

  if(NOT EXISTS ${blas_library_path})
    message(FATAL_ERROR "   cblas library was not found: ${blas_library_path}")
  endif()
  
  if(NOT EXISTS ${blas_headers_path})
    message(FATAL_ERROR "   cblas headers not found in: ${blas_headers_path}")
  endif()
    
  set_target_properties(cblas
    PROPERTIES
      IMPORTED_LOCATION ${blas_library_path}
      INTERFACE_INCLUDE_DIRECTORIES ${blas_headers_path})
else()
  # For Linux and macOS, we provide the .a file
  
  # Check if the required files and paths exist
  set(blas_library_path "${CMAKE_CURRENT_LIST_DIR}/SuperLU_5.2.1/libblas.a")  # set the location of blas library file (full path)
  set(blas_headers_path "${CMAKE_CURRENT_LIST_DIR}/SuperLU_5.2.1/SRC")  # set the location of blas header files 
  # set(blas_header_file_to_check "slu_ddefs.h")  # check for this header file inside the header files  TODO: Need to find a header file that is relevant

  if(NOT EXISTS ${blas_library_path})
    message(FATAL_ERROR "   cblas library was not found: ${blas_library_path}")
  endif()
  
  if(NOT EXISTS ${blas_headers_path})
    message(FATAL_ERROR "   cblas headers not found in: ${blas_headers_path}")
  endif()
  
  set_target_properties(cblas
    PROPERTIES
      IMPORTED_LOCATION ${blas_library_path}
      INTERFACE_INCLUDE_DIRECTORIES ${blas_headers_path})
endif()

# Finalize reporting and check if all libraries have been added ----------------
message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------