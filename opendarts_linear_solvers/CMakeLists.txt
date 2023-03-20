cmake_minimum_required (VERSION 3.21)
project (opendarts_linear_solvers LANGUAGES CXX VERSION 0.2)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Load cmake modules -----------------------------------------------------------
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/settings/cmake/modules/")

include(FeatureSummary)
include(GetGitRevisionDescription)  # to extract git information (availabe in source)
include(BundleCMakeStaticLibs)  # to bundle a static lib and its dependencies into a single static lib
# ------------------------------------------------------------------------------


# Get environment information for the build and IDEs ---------------------------

# Give the user some feedback on what is going on
message(CHECK_START "Defining build settings")  # build settings

# Get hostname and username 
cmake_host_system_information(RESULT opendarts_linear_solvers_BUILD_HOSTNAME QUERY HOSTNAME)
if(WIN32)
  set(opendarts_linear_solvers_BUILD_USERNAME $ENV{USERNAME})
else()
  set(opendarts_linear_solvers_BUILD_USERNAME $ENV{USER})
endif()

# Show build information 
message(STATUS "  username: ${opendarts_linear_solvers_BUILD_USERNAME}")
message(STATUS "  hostname: ${opendarts_linear_solvers_BUILD_HOSTNAME}")

# Get git information
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
git_local_changes(GIT_DIRTY)

# Show git information 
message(STATUS "  git refspec: ${GITREFSPEC}")
message(STATUS "  git hash: ${GIT_SHA1}")
message(STATUS "  git dirtiness: ${GIT_DIRTY}")

# Determine the type of compiler we have and set compiler flags and other compiler 
# specific flags

# Set compiler specific variables
# Used for compiler flags and other cross compiler variables
message(STATUS "  Setting compiler and linker flags for: ${CMAKE_CXX_COMPILER_ID}")
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # Compiler flags 
  add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)
  set(opendarts_cxx_compiler_flags, "-W3")
  
  # Linker flags (currently not needed)
  # set(opendarts_cxx_linker_flags, "-W3")
  
  # Compiler specific helper variables 
   
  # Flags cmake to import external libraries as msvc like files: .lib and .dll
  # instead of .a and .so
  set(import_externals_as_msvc TRUE)
  
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "  Setting GNU compiler and linker flags")
  # Compiler flags 
  set(opendarts_cxx_compiler_flags "-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused;-fPIC")
  
  if(SET_CXX11_ABI_0)
    list(APPEND opendarts_cxx_compiler_flags "-D_GLIBCXX_USE_CXX11_ABI=0")
  endif()
  
  # Linker flags (currently not needed)
  # set(opendarts_cxx_linker_flags, "-W3")
  
  # Compiler specific helper variables (currently not needed)

elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(STATUS "  Setting Clang compiler and linker flags")
  # Compiler flags 
  set(opendarts_cxx_compiler_flags "-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused;-fPIC")
  
  if(SET_CXX11_ABI_0)
    list(APPEND opendarts_cxx_compiler_flags "-D_GLIBCXX_USE_CXX11_ABI=0")
  endif()
  
  # Linker flags
  # Reasons:
  #   -Wl,-no_compact: used because in Clang compilers we get linker warning of
  #                    being unable to create compact unwinding. This is relevant 
  #                    for efficient exceptions, but this comes from SuperLU, so 
  #                    we deactivate compact unwinding warnings.  
  set(opendarts_cxx_linker_flags "-Wl,-no_compact_unwind")
  
  # Compiler specific helper variables (currently not needed)
  
endif()

message(STATUS "   flags: ${opendarts_cxx_compiler_flags}")

# Setup options
option(ENABLE_TESTING "Build and copy testing stuff" OFF)  # by default tests are not built, to save time
add_feature_info(ENABLE_TESTING ENABLE_TESTING "Build and copy testing stuff")  # add option as feature for pretty display

# Report options for this build
feature_summary(WHAT ENABLED_FEATURES DISABLED_FEATURES PACKAGES_FOUND)
feature_summary(FILENAME ${CMAKE_CURRENT_BINARY_DIR}/features.log WHAT ALL)
message(STATUS "The list of features was also saved to file: ${CMAKE_CURRENT_BINARY_DIR}/features.log\n")

# Allows setting folders in IDE and placing targets there, otherwise not possible
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

message(CHECK_PASS "done!")  # build settings
# ------------------------------------------------------------------------------


# Setup testing ----------------------------------------------------------------

# Report Enable or disable testing via option flag
if(ENABLE_TESTING)
  enable_testing()
endif()
# ------------------------------------------------------------------------------


# Build dummy library to contain compiler flags --------------------------------
add_library(opendarts_compiler_flags INTERFACE)

target_compile_features(opendarts_compiler_flags
  INTERFACE
  cxx_std_11
)

target_compile_options(opendarts_compiler_flags 
  INTERFACE
    ${opendarts_cxx_compiler_flags}
)

target_link_options(opendarts_compiler_flags 
  INTERFACE
    ${opendarts_cxx_linker_flags}
)

# ------------------------------------------------------------------------------


# Include thirdparty libraries -------------------------------------------------
add_subdirectory(thirdparty)
# ------------------------------------------------------------------------------


# Include config library -------------------------------------------------------
add_subdirectory(config)
# ------------------------------------------------------------------------------


# Include auxiliary library ----------------------------------------------------
add_subdirectory(auxiliary)
# ------------------------------------------------------------------------------


# Include linear_solvers library -----------------------------------------------
add_subdirectory(linear_solvers)
# ------------------------------------------------------------------------------


# Include the tests ------------------------------------------------------------
if(ENABLE_TESTING)
  add_subdirectory(tests)
endif()
# ------------------------------------------------------------------------------
