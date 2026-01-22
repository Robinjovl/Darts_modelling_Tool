# MGR Linear Solver - Integrated into open-darts build system

# IMPORTANT: Now using open-darts's rebuilt HYPRE library with MGR support
# The HYPRE library is rebuilt from source (2.29.0) to ensure MGR functionality
# See compile_and_test_mgr.bat Step 0 for HYPRE rebuild process

# Source files
set(LIB_SOURCES
    src/LinearSolver.cpp
    src/MGRStrategy.cpp
    src/CompositionalFlowStrategy.cpp
)

# Create library
add_library(mgr-linear-solver ${LIB_SOURCES})

# Build as PIC so it can link into shared objects (Python extensions, etc.)
if(NOT WIN32)
    set_target_properties(mgr-linear-solver PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Include directories
target_include_directories(mgr-linear-solver PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link HYPRE (using open-darts's HYPRE::HYPRE target - now with MGR support)
target_link_libraries(mgr-linear-solver PUBLIC HYPRE::HYPRE)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(mgr-linear-solver PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Alias for easier linking
add_library(opendarts::mgr-linear-solver ALIAS mgr-linear-solver)

message(STATUS "MGR Linear Solver configured")

# ========================================================================
# open-darts Integration: LinsolvIfaceMGR
# ========================================================================

# Source files for open-darts integration
set(INTEGRATION_SOURCES
    src/LinsolvIfaceMGR.cpp
)

# Create integration library (static library for MGR solver + open-darts interface)
add_library(linsolv_mgr ${INTEGRATION_SOURCES})

# Ensure PIC for integration library as well
if(NOT WIN32)
    set_target_properties(linsolv_mgr PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Include directories
# Use absolute paths to avoid relative path resolution issues
target_include_directories(linsolv_mgr PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/solvers/linear_solvers/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/solvers/auxiliary/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/solvers/config/include>
)

# Link with mgr-linear-solver library and config/auxiliary
target_link_libraries(linsolv_mgr PUBLIC
    mgr-linear-solver
    config
    auxiliary
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(linsolv_mgr PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Alias for easier linking in open-darts
add_library(opendarts::linsolv_mgr ALIAS linsolv_mgr)

message(STATUS "MGR Integration (LinsolvIfaceMGR) configured")

# ========================================================================
# Test Executable (DISABLED - MGR is now tested through open-darts examples)
# ========================================================================

# Test executable disabled - file I/O functions removed from LinearSolver
# MGR solver is tested through open-darts examples (e.g., models/2ph_comp)
# To enable file-based testing, restore BlockCSRReader and add loadSystem/exportToCSV methods
