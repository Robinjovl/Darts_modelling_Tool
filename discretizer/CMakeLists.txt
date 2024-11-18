# Give the user some feedback on what is going on
message(STATUS "Building openDARTS discretizer")  

# Get hostname and username 
cmake_host_system_information(RESULT opendarts_discretizer_BUILD_HOSTNAME QUERY HOSTNAME)
if(WIN32)
  set(opendarts_discretizer_BUILD_USERNAME $ENV{USERNAME})
else()
  set(opendarts_discretizer_BUILD_USERNAME $ENV{USER})
endif()

# Thirdparty libraries ---------------------------------------------------------
# Reports we started looking for MShIO libraries and initializes the flag 
# that checks if all have been found 
message(CHECK_START "   Importing MshIO")
if(NOT DEFINED MshIO_DIR)
  set(MshIO_DIR "${CMAKE_CURRENT_LIST_DIR}/../thirdparty/MshIO")
endif(NOT DEFINED MshIO_DIR)

file(GLOB MSHIO_INC_FILES "${MshIO_DIR}/include/mshio/*.h")
file(GLOB MSHIO_SRC_FILES "${MshIO_DIR}/src/*.cpp")

add_library(mshio OBJECT "")

target_sources(mshio
  PUBLIC
    ${MSHIO_SRC_FILES}
    ${MSHIO_INC_FILES}  # if added as PUBLIC appears in all targets that link to it
)

target_include_directories(mshio PUBLIC "${MshIO_DIR}/include/" )

set(mshio_cxx_compiler_flags "-fPIC;-std=c++17;-fno-lto")
target_compile_options(mshio 
  PRIVATE
    ${mshio_cxx_compiler_flags}
)

if (TARGET mshio)
  message(STATUS "      Found MshIO: TRUE")
else()
  message(FATAL_ERROR "      Found MshIO: FALSE")
endif (TARGET mshio)

# Some user feedback info
# check that is changed correctly
get_target_property(MSHIO_INCLUDE_DIRS mshio INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "      MshIO_DIR: ${MshIO_DIR}")
message(STATUS "      MshIO Include directories: ${MSHIO_INCLUDE_DIRS}")

# Finalize reporting and check if all libraries have been added ----------------
message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------


# Discretizer ------------------------------------------------------------------
add_subdirectory(src)
# ------------------------------------------------------------------------------

message(STATUS "Building openDARTS discretizer - done!")  