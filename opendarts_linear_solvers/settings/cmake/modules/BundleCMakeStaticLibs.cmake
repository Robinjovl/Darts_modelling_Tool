# Source: https://stackoverflow.com/questions/37924383/combining-several-static-libraries-into-one-using-cmake

# - Bundles static libraries into a single static library
#      add_library(awesome_lib STATIC ...);  # containing depedencies of other static libs
#      set(bundled_dir "bundled_libs_foldername")  # the foldername where to store temporary bundled libs
#      bundle_static_library(awesome_lib bundled_dir bundled_tgt_full_file_name bundled_tgt_include_headers)
#   The function bundle_static_library generates a single file containing all 
#   dependences in 
#      <CMAKE_BINARY_DIR>/${bundled_dir} 
#   named
#      libawesome_lib.a 
#   returns the full path to that file in bundled_tgt_full_file_name
#   returns the list of include header file directories required to use the library 
#   in bundled_tgt_include_headers.
#
function(bundle_static_library tgt_name bundled_dir bundled_tgt_full_file_name bundled_tgt_file_name bundled_tgt_include_headers bundling_target)
  message(CHECK_START "Bundling ${tgt_name}...")  # build settings
  
  list(APPEND static_libs ${tgt_name})  # contains the list of all static libs tgt_name depends upon 
  list(APPEND object_libs ${tgt_name})  # contains the list of all object libs tgt_name depends upon
  
  message(STATUS "   Finding dependencies")
  function(_recursively_collect_dependencies input_target)
    set(_input_link_libraries LINK_LIBRARIES)
    get_target_property(_input_type ${input_target} TYPE)
    if (${_input_type} STREQUAL "INTERFACE_LIBRARY")
      set(_input_link_libraries INTERFACE_LINK_LIBRARIES)
    endif()
    get_target_property(public_dependencies ${input_target} ${_input_link_libraries})
    
    foreach(dependency IN LISTS public_dependencies)
      if(TARGET ${dependency})
        get_target_property(alias ${dependency} ALIASED_TARGET)
        if (TARGET ${alias})
          set(dependency ${alias})
        endif()
        get_target_property(_type ${dependency} TYPE)
        if (${_type} STREQUAL "STATIC_LIBRARY")
          list(APPEND static_libs ${dependency})
        endif()
        if (${_type} STREQUAL "OBJECT_LIBRARY")
          list(APPEND object_libs ${dependency})
        endif()
        
        get_property(library_already_added
          GLOBAL PROPERTY _${tgt_name}_static_bundle_${dependency})
        if (NOT library_already_added)
          set_property(GLOBAL PROPERTY _${tgt_name}_static_bundle_${dependency} ON)
          _recursively_collect_dependencies(${dependency})
        endif()
      endif()
    endforeach()
    set(static_libs ${static_libs} PARENT_SCOPE)
    set(object_libs ${object_libs} PARENT_SCOPE)
  endfunction()

  _recursively_collect_dependencies(${tgt_name})

  list(REMOVE_DUPLICATES static_libs)
  list(REMOVE_DUPLICATES object_libs)
  
  message(STATUS "   Setting up bundling")
  set(bundled_tgt_full_name 
  ${CMAKE_BINARY_DIR}/${bundled_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_tgt_file_name}${CMAKE_STATIC_LIBRARY_SUFFIX})
  
  if ((CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|GNU)$") AND NOT APPLE)
    message(STATUS "      linux bundling")
    file(WRITE ${CMAKE_BINARY_DIR}/${bundled_dir}/${bundled_tgt_file_name}.ar.in
      "CREATE ${bundled_tgt_full_name}\n" )
        
    foreach(tgt IN LISTS static_libs)
      file(APPEND ${CMAKE_BINARY_DIR}/${bundled_dir}/${bundled_tgt_file_name}.ar.in
        "ADDLIB $<TARGET_FILE:${tgt}>\n")
    endforeach()
    
    file(APPEND ${CMAKE_BINARY_DIR}/${bundled_dir}/${bundled_tgt_file_name}.ar.in "SAVE\n")
    file(APPEND ${CMAKE_BINARY_DIR}/${bundled_dir}/${bundled_tgt_file_name}.ar.in "END\n")

    file(GENERATE
      OUTPUT ${CMAKE_BINARY_DIR}/${bundled_dir}/${bundled_tgt_file_name}.ar
      INPUT ${CMAKE_BINARY_DIR}/${bundled_dir}/${bundled_tgt_file_name}.ar.in)

    set(ar_tool ${CMAKE_AR})
    if (CMAKE_INTERPROCEDURAL_OPTIMIZATION)
      set(ar_tool ${CMAKE_CXX_COMPILER_AR})
    endif()

    add_custom_command(
      COMMAND ${ar_tool} -M < ${CMAKE_BINARY_DIR}/${bundled_dir}/${bundled_tgt_file_name}.ar
      OUTPUT ${bundled_tgt_full_name}
      COMMENT "Bundling ${bundled_tgt_file_name}"
      VERBATIM)
      
  elseif ((CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang|GNU)$") AND APPLE)
    message(STATUS "      macOS bundling")
    set(ar_tool ${CMAKE_AR})
    if (CMAKE_INTERPROCEDURAL_OPTIMIZATION)
      set(ar_tool ${CMAKE_CXX_COMPILER_AR})
    endif()
    
    foreach(tgt IN LISTS static_libs)
      list(APPEND static_libs_full_names $<TARGET_FILE:${tgt}>)
    endforeach()
    
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${bundled_dir})
    add_custom_command(
      COMMAND libtool -static -o ${bundled_tgt_full_name} ${static_libs_full_names}
      OUTPUT ${bundled_tgt_full_name}
      COMMENT "Bundling ${bundled_tgt_file_name}"
      VERBATIM)
      
  elseif(MSVC)
    message(STATUS "      MSVC bundling")
    find_program(lib_tool lib)

    foreach(tgt IN LISTS static_libs)
      list(APPEND static_libs_full_names $<TARGET_FILE:${tgt}>)
    endforeach()

    add_custom_command(
      COMMAND ${lib_tool} /NOLOGO /OUT:${bundled_tgt_full_name} ${static_libs_full_names}
      OUTPUT ${bundled_tgt_full_name}
      COMMENT "Bundling ${bundled_tgt_file_name}"
      VERBATIM)
      
  else()
    message(FATAL_ERROR "      Unknown bundle scenario!")
  endif()

  add_custom_target(${bundling_target} ALL DEPENDS ${bundled_tgt_full_name})
  add_dependencies(${bundling_target} ${tgt_name})
  
  message(STATUS "   Collecting include header dirs")
  foreach(dependency IN LISTS object_libs)
    message(STATUS "      ${dependency}")
    get_target_property(LIBB_INCLUDES ${dependency} INCLUDE_DIRECTORIES)
    foreach(dir ${LIBB_INCLUDES})
      if(${dir} MATCHES ".*BUILD_INTERFACE")
        string(REPLACE "$<BUILD_INTERFACE:" "" dir ${dir})
        string(REPLACE ">" "" dir ${dir})
        list(APPEND include_headers_lists ${dir})
        message(STATUS "         ${dir}")
      endif()
    endforeach()
  endforeach()
  
  foreach(dependency IN LISTS static_libs)
    message(STATUS "      ${dependency}")
    get_target_property(LIBB_INCLUDES ${dependency} INTERFACE_INCLUDE_DIRECTORIES)
    foreach(dir ${LIBB_INCLUDES})
      if(${dir} MATCHES ".*BUILD_INTERFACE")
        string(REPLACE "$<BUILD_INTERFACE:" "" dir ${dir})
        string(REPLACE ">" "" dir ${dir})
        list(APPEND include_headers_lists ${dir})
        message(STATUS "         ${dir}")
      elseif(${dir} MATCHES ".*INSTALL_INTERFACE")
      else()
        list(APPEND include_headers_lists ${dir})
        message(STATUS "         ${dir}")
      endif()
    endforeach()
  endforeach()
  
  list(REMOVE_DUPLICATES include_headers_lists)
  
  # message("include_headers_lists_STR: ${include_headers_lists}")
  message(STATUS "   Bundled file")
  message(STATUS "         ${bundled_tgt_full_name}")
  
  set(${bundled_tgt_include_headers} ${include_headers_lists} PARENT_SCOPE)
  set(${bundled_tgt_full_file_name} ${bundled_tgt_full_name} PARENT_SCOPE)
  
  message(CHECK_PASS "done!")  # build settings
endfunction()
