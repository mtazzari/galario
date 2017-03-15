cmake_minimum_required(VERSION 3.0)
project(galario)
set(PACKAGE_VERSION 0.1)

# ship extra findXXX.cmake in `cmake/`
# APPEND doesn't propagate variable to subdirectory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
# set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
include(UseCython)

# With CMake, a clean separation can be made between the source tree and the
# build tree.  When all source is compiled, as with pure C/C++, the source is
# no-longer needed in the build tree.  However, with pure *.py source, the
# source is processed directly.  To handle this, we reproduce the availability
# of the source files in the build tree.
# add_custom_target( ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
#   ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplicatePythonSourceTree.cmake
#   ${CMAKE_CURRENT_BINARY_DIR}
#   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python )

# we can't use static libs with python
set(BUILD_SHARED_LIBS ON)
add_subdirectory(src)
add_subdirectory(python)
