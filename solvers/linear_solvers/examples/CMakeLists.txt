
# Adds all examples. To do that, it is assumed that each .cpp file is an example
# and the example's name is the filename without the extension.

# Collect all the names of the cpp files
file(GLOB CPPFILES  "*.cpp")

# For every cpp found
foreach(CPPFILE ${CPPFILES})
	# Extract the actual file name
	get_filename_component(FILENAME ${CPPFILE} NAME)

	# Extract the filename minus the cpp. This will be the name of target
	get_filename_component(TARGETNAME ${CPPFILE} NAME_WE)

	# Make the target
	add_executable(${TARGETNAME} ${FILENAME})

	target_link_libraries(${TARGETNAME}
    config
    opendarts_compiler_flags
    linear_solvers
  )

  # Place the target test in the examples folder and rename the source folder to src
  set_property(TARGET ${TARGETNAME} PROPERTY FOLDER "examples//linear_solvers")
  source_group(src FILES ${FILENAME})

	# Install the example executable
	install(TARGETS ${TARGETNAME} DESTINATION bin/openDARTS/examples/linear_solvers)
endforeach()
