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


# Adds linear_solvers ----------------------------------------------------------
message(CHECK_START "   Importing linear_solvers")

add_library(linear_solvers STATIC IMPORTED GLOBAL)
# When setting the library files to import, we need to do it in different ways for
# Windows and linux/macOS
if(import_externals_as_msvc)
  # For Windows we need to set the .lib file, which has a different name from the .a file
  
  # Check if the required files and paths exist
  set(linear_solvers_library_path "${CMAKE_CURRENT_SOURCE_DIR}/solvers/lib/darts_linear_solvers.lib")  # set the location of linear_solvers library (full path)
  set(linear_solvers_headers_path "${CMAKE_CURRENT_SOURCE_DIR}/solvers/include")  # set the location of linear_solvers header files
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
  set(linear_solvers_library_path "${CMAKE_CURRENT_SOURCE_DIR}/solvers/lib/libdarts_linear_solvers.a")  # set the location of linear_solvers library file (full path)
  set(linear_solvers_headers_path "${CMAKE_CURRENT_SOURCE_DIR}/solvers/include")  # set the location of linear_solvers header files 
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


# # Adds python ------------------------------------------------------------------
# message(CHECK_START "   Importing python3")

# find_package (Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# # # Check if python3 was found
# # if(NOT Python3_FOUND)
# #   message(FATAL_ERROR "      python3 not found!")
# # endif()
# # 
# # # Check if development module was found
# # if(NOT Python3_Development.Module_FOUND)
# #   message(FATAL_ERROR "   python3 Development.Module not found! You may need to intall python development tools.")
# # endif()

# # Some user feedback info
# message(STATUS "      Python version: ${Python3_VERSION}")
# message(STATUS "      Include directory: ${Python3_INCLUDE_DIRS}")
# message(STATUS "      Additional Link options: ${Python3_LINK_OPTIONS}")

# message(CHECK_PASS "done!")
# # ------------------------------------------------------------------------------


# # Adds pybind11 ----------------------------------------------------------------
# message(CHECK_START "   Importing pybind11")

# # add_library(pybind11 INTERFACE)
# # target_include_directories(pybind11 INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/pybind11")

# add_subdirectory(pybind11_new)
# # Luisa added:
# message(STATUS "      Include directories: ${pybind11_INCLUDE_DIRS}")
# if (TARGET  pybind11::module)
#   message(STATUS "      Found pybind11: TRUE")
# else()
#   message(FATAL_ERROR "      Found pybind11: FALSE")
# endif (TARGET  pybind11::module)
# # end Luisa added

# message(CHECK_PASS "done!")
# # ------------------------------------------------------------------------------

# Finalize reporting and check if all libraries have been added ----------------
message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------
