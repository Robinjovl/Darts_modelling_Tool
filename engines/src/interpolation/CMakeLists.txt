# engines_interpolator — thin INTERFACE wrapper around darts_interpolation ------
# All source files have moved to the top-level interpolation/ library.
# This target preserves the old name so existing link dependencies keep working.

add_library(engines_interpolator INTERFACE)
target_link_libraries(engines_interpolator INTERFACE darts_interpolation engines_global)

# Expose interpolation pybind11 headers so engine_base.h -> py_globals.h ->
# py_globals_interpolation.h chain resolves for all consumers.
target_include_directories(engines_interpolator
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/interpolation/pybind11>
)

# ------------------------------------------------------------------------------
