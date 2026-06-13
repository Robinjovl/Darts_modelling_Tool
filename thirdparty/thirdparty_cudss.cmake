# NVIDIA cuDSS -- GPU sparse direct solver (https://docs.nvidia.com/cuda/cudss/).
#
# Unlike AMGX, cuDSS ships PREBUILT (header + shared library); there is no
# from-source submodule build. Acquisition options (any one):
#   * pip wheel into the active env:  pip install nvidia-cudss-cu13
#     (lands in <site-packages>/nvidia/cu13/{include,lib})
#   * NVIDIA redistributable tarball: set CUDSS_ROOT=<extracted dir>
#   * a system/conda package providing cudss-config.cmake on CMAKE_PREFIX_PATH
#
# Resolution order:
#   1. find_package(cudss CONFIG) -- honours CMAKE_PREFIX_PATH / cudss_DIR.
#   2. Manual find_path/find_library with hints: CUDSS_ROOT (cache or env),
#      then the active conda/virtualenv's nvidia/cu* wheel layout.
# Defines the imported target `cudss::cudss` either way.

if(TARGET cudss::cudss)
  return()
endif()

if(NOT CUDSS_ROOT AND DEFINED ENV{CUDSS_ROOT})
  set(CUDSS_ROOT "$ENV{CUDSS_ROOT}")
endif()

find_package(cudss CONFIG QUIET HINTS "${CUDSS_ROOT}")
if(cudss_FOUND AND TARGET cudss)
  # NVIDIA's config file exports a bare `cudss` target; normalise the name.
  add_library(cudss::cudss ALIAS cudss)
  message(STATUS "  cuDSS: found via CONFIG package (${cudss_DIR})")
  return()
endif()

# Manual discovery -- pip-wheel layout inside the active Python environment
# (e.g. mambaforge env `solvers`: .../site-packages/nvidia/cu13/{include,lib}),
# then CUDSS_ROOT.
set(_cudss_hint_dirs "")
if(CUDSS_ROOT)
  list(APPEND _cudss_hint_dirs "${CUDSS_ROOT}")
endif()
foreach(_env_root "$ENV{CONDA_PREFIX}" "$ENV{VIRTUAL_ENV}")
  if(_env_root)
    file(GLOB _wheel_dirs "${_env_root}/lib/python*/site-packages/nvidia/cu*")
    list(APPEND _cudss_hint_dirs ${_wheel_dirs})
  endif()
endforeach()

find_path(CUDSS_INCLUDE_DIR cudss.h
  HINTS ${_cudss_hint_dirs}
  PATH_SUFFIXES include)
# pip wheels ship only the versioned soname (libcudss.so.0) -- search for it
# explicitly alongside the regular name.
find_library(CUDSS_LIBRARY
  NAMES cudss libcudss.so.0
  HINTS ${_cudss_hint_dirs}
  PATH_SUFFIXES lib lib64)

if(NOT CUDSS_INCLUDE_DIR OR NOT CUDSS_LIBRARY)
  # WITH_CUDSS is ON by default for GPU builds; a missing prebuilt library must
  # NOT hard-fail the whole GPU build. Warn, disable cuDSS, and continue -- the
  # GPU stack is then built without the cuDSS direct solver. (Setting the cache
  # value with FORCE propagates the disable to solvers/src/CMakeLists.txt.)
  message(WARNING
    "WITH_CUDSS=ON but cuDSS was not found -- building the GPU stack WITHOUT cuDSS.\n"
    "  To enable it, provide cuDSS via one of:\n"
    "    pip install nvidia-cudss-cu13   (into the build's Python env)\n"
    "    -D CUDSS_ROOT=<dir>             (NVIDIA redistributable layout)\n"
    "    CMAKE_PREFIX_PATH containing cudss-config.cmake\n"
    "  Pass -D WITH_CUDSS=OFF to silence this warning. Searched hints: ${_cudss_hint_dirs}")
  set(WITH_CUDSS OFF CACHE BOOL
    "Build the cuDSS GPU direct solver (prebuilt NVIDIA library)" FORCE)
  unset(_cudss_hint_dirs)
  return()
endif()

add_library(cudss_imported SHARED IMPORTED GLOBAL)
set_target_properties(cudss_imported PROPERTIES
  IMPORTED_LOCATION "${CUDSS_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${CUDSS_INCLUDE_DIR}")
add_library(cudss::cudss ALIAS cudss_imported)

# Bundle the runtime library into the darts/ install tree: the Python
# extension modules are linked with RPATH=$ORIGIN, so a copy of
# libcudss.so.0 next to engines.so resolves without LD_LIBRARY_PATH
# (mirrors the libamgxsh / IPhreeqc bundling).
install(FILES "${CUDSS_LIBRARY}" DESTINATION "${CMAKE_INSTALL_PREFIX}")

message(STATUS "  cuDSS: ${CUDSS_LIBRARY} (includes: ${CUDSS_INCLUDE_DIR})")
unset(_cudss_hint_dirs)
