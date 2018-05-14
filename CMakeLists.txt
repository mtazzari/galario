cmake_minimum_required(VERSION 3.0)
project(galario)
set(PACKAGE_VERSION 1.03)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
  endif(NOT CMAKE_BUILD_TYPE)

# The NDEBUG turns off error printing but we want that even in the release
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

# if inside a conda env, install everything in there
# if (DEFINED ENV{CONDA_PREFIX} AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
if (DEFINED ENV{CONDA_PREFIX})
  message(STATUS "Installing inside conda env at $ENV{CONDA_PREFIX}")
  set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
endif()

# ship extra findXXX.cmake in `cmake/`
# APPEND doesn't propagate variable to subdirectory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# many more useful cmake module and scripts
# are downloaded into the build directory
include(LookUp-GreatCMakeCookOff)

###
# the optional packages
###
if(NOT DEFINED GALARIO_CHECK_CUDA)
  if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    message(STATUS "Ignoring cuda on the mac by default. Force searching for cuda with `cmake -DGALARIO_CHECK_CUDA=1`")
    set(GALARIO_CHECK_CUDA 0)
  else()
    set(GALARIO_CHECK_CUDA 1)
  endif()
endif()
if(GALARIO_CHECK_CUDA)
  find_package(CUDA)
endif()

# has to come before subdirectories are added
enable_testing()

# we can't use static libs with python
set(BUILD_SHARED_LIBS ON)
add_subdirectory(src)

###
# python not required but optional
###
set(BUILD_WRAPPER FALSE)

# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
find_package(Cython)
if (CYTHON_EXECUTABLE)
  include(UseCython)
endif()

if(PYTHONLIBS_FOUND AND CYTHON_FOUND)
  # to avoid fatal error: numpy/arrayobject.h: No such file or directory
  find_package(Numpy)

  # should check NUMPY_FOUND but there is a bug in GreatCmakeCookoff
  # https://github.com/UCL/GreatCMakeCookOff/issues/57
  # upon calling `make` repeatedly, NUMPY_FOUND set to false
  if (NUMPY_INCLUDE_DIRS)
    set(BUILD_WRAPPER TRUE)
    include_directories("${NUMPY_INCLUDE_DIRS}")
  else ()
    message(STATUS "NumPy not found")
  endif()
endif()

if(BUILD_WRAPPER)
  message(STATUS "Building the ${PROJECT_NAME} python wrapper")
  add_subdirectory(python)
else()
  message(STATUS "Skipping the ${PROJECT_NAME} python wrapper")
endif()

find_package(Sphinx)
if(SPHINX_FOUND)
  add_subdirectory(docs)
else()
  message(STATUS "Cannot build the documentation without sphinx")
endif()

###
# uninstall target
###
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
