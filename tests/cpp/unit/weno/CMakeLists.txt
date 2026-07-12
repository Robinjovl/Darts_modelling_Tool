add_executable(discretizer_weno_static weno_static.cpp)

target_link_libraries(discretizer_weno_static
  opendarts_compiler_flags
  opendarts_discretizer
)

set_property(TARGET discretizer_weno_static PROPERTY FOLDER "tests//unit//weno")
add_test(NAME discretizer_weno_static COMMAND discretizer_weno_static)
