cmake_minimum_required(VERSION 3.0)
project(galario)
set(PACKAGE_VERSION 0.1)

# ship extra findXXX.cmake in `cmake/`
# APPEND doesn't propagate variable to subdirectory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# many more useful cmake module and scripts
# are downloaded into the build directory
include(LookUp-GreatCMakeCookOff)

###
# the optional and required packages
###
find_package(CUDA)

# User can change and specify version of python in case there are
# several in cache with `cmake-gui .` or `ccmake .`
# set(Python_ADDITIONAL_VERSIONS 3;2 CACHE STRING "List of python versions to look for. First come, first served, for example '2;3.5' prefers 2.x over 3.5")

# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
include(UseCython)

# has to come before subdirectories are added
enable_testing()

# we can't use static libs with python
set(BUILD_SHARED_LIBS ON)
add_subdirectory(src)

# TODO perhaps python not required but optional?

# if(PYTHONLIBS_FOUND)
add_subdirectory(python)
# endif()
