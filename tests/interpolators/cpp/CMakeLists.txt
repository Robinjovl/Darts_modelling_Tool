
# Adds all tests. To do that, it is assumed that each .cpp file is a test and the
# test's name is the filename without the extension.

# Collect all the names of the cpp files
file(GLOB CPPFILES  "*.cpp")

# For every cpp found
foreach(CPPFILE ${CPPFILES})
	# Extract the actual file name
	get_filename_component(FILENAME ${CPPFILE} NAME)

	# Extract the filename minus the cpp. This will be the base name of target
	get_filename_component(TARGETNAME_BASE ${CPPFILE} NAME_WE)

	# To distinguish tests from different components we add a prefix
	set(TARGETNAME "interpolators_${TARGETNAME_BASE}")


	# Make the target
	add_executable(${TARGETNAME} ${FILENAME})

	target_link_libraries(${TARGETNAME}
    opendarts_compiler_flags
  )

  target_include_directories(${TARGETNAME}
    PRIVATE
      ${CMAKE_SOURCE_DIR}/interpolation/include
  )

  # Place the target test in the tests folder and rename the source folder to src
  set_property(TARGET ${TARGETNAME} PROPERTY FOLDER "tests//interpolators")
  source_group(src FILES ${FILENAME})

	# Add the test
	add_test(NAME ${TARGETNAME} COMMAND ${TARGETNAME})
endforeach()
