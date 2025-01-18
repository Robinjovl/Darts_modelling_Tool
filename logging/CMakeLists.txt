cmake_minimum_required(VERSION 3.21)
project(logging) 

# -- C++ logging
# Define the logging library
add_library(opendarts_logging OBJECT
  ${CMAKE_CURRENT_SOURCE_DIR}/src/logging.cpp)


# Define the include directories to propagate when linking
target_include_directories(opendarts_logging
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(opendarts_logging PRIVATE
  opendarts_compiler_flags
)

install(TARGETS opendarts_logging 
  CONFIGURATIONS RELEASE    
  DESTINATION lib
)

install(TARGETS opendarts_logging
  CONFIGURATIONS DEBUG     
  DESTINATION lib
)

# -- End C++ logging


# -- Python binding via pybind11
pybind11_add_module(logging 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/py_logging.cpp
)

# Link pybind11 to the logging module
target_link_libraries(logging PRIVATE
  opendarts_compiler_flags
  opendarts_logging 
  pybind11::module)

install(TARGETS logging 
  CONFIGURATIONS RELEASE RELWITHDEBINFO DEBUG ""
  DESTINATION ./
)
