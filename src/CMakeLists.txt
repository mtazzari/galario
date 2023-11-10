#####################################################################################
# This file is part of GALARIO.                                                     #
#                                                                                   #
# GALARIO - Gpu Accelerated Library for Analysing Radio Interferometer Observations #
# Copyright (C) 2017-2020, Marco Tazzari, Frederik Beaujean, Leonardo Testi.        #
#                                                                                   #
# This program is free software: you can redistribute it and/or modify              #
# it under the terms of the Lesser GNU General Public License as published by       #
# the Free Software Foundation, either version 3 of the License, or                 #
# (at your option) any later version.                                               #
#                                                                                   #
# This program is distributed in the hope that it will be useful,                   #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                    #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                              #
#                                                                                   #
# For more details see the LICENSE file.                                            #
# For documentation see https://mtazzari.github.io/galario/                         #
#####################################################################################

# C++ 11 support http://stackoverflow.com/questions/37621342/ for cmake >= 2.8
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  set(min_gcc 4.8.1)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${min_gcc})
    message(FATAL_ERROR "${PROJECT_NAME} requires g++ >= ${min_gcc} for C++11 support.")
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(min_clang 3.3)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${min_clang})
    message(FATAL_ERROR "${PROJECT_NAME} requires clang++ >= ${min_clang} for C++11 support.")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  endif()
else()
  message(FATAL_ERROR "Your compiler is not supported for lack of complete C++11 support")
endif()

# these options are understood by both gcc and nvcc. The latter takes into account ${CMAKE_CXX_FLAGS}
set(cxx_std "-std=c++11")
set(CMAKE_CXX_FLAGS "${cxx_std} ${CMAKE_CXX_FLAGS}")

# these need to be wrapped in -Xcompiler for nvcc
# use strict-aliasing because all input arrays to functions must not alias for correctness
set(extra_compile_options  -Wall -pedantic -fstrict-aliasing -pthread)
add_compile_options(${extra_compile_options})

# we compile the same source file multiple times, varying preprocessor definitions
set(common_cpp "${CMAKE_CURRENT_SOURCE_DIR}/galario.cpp")

# compile single and double precision in separate libraries
add_library(galario_single ${common_cpp})

add_library(galario ${common_cpp})
target_compile_definitions(galario PUBLIC DOUBLE_PRECISION)

set(install_libs galario_single galario)

find_package(OpenMP)
find_package(Threads)
if(OpenMP_FOUND AND Threads_FOUND)
  set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
  # https://cmake.org/Bug/view.php?id=9075
  # http://stackoverflow.com/questions/24532853/how-to-add-linker-flag-for-libraries-with-cmake
  set(CMAKE_SHARED_LINKER_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
  set(FFTW_PARALLEL "THREADS")
endif()

# we need fftw with (single+double)x(serial+openmp)
find_package(FFTW3 COMPONENTS SINGLE DOUBLE ${FFTW_PARALLEL})

OPTION(GALARIO_TIMING "Output timing of selected functions. For testing only. Default: OFF" OFF)

foreach(t IN ITEMS galario_single galario)
  target_link_libraries(${t} ${FFTW3_LIBRARIES})
  target_include_directories(${t} PUBLIC ${FFTW3_INCLUDE_DIRS} ${FFTW3_INCLUDE_DIR_PARALLEL} ../delaunator-cpp/include)
  if(GALARIO_TIMING)
    target_compile_definitions(${t} PRIVATE GALARIO_TIMING=1)
  endif()
endforeach()

if (CUDA_FOUND)
  # We require at least 30 because anything before is deprecated in cuda 8. For
  # maximum compatibility with future GPUs, we don't specify an exact GPU code
  # like such as sm_53. The code is JIT compiled incurring a startup delay.
  # http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#further-mechanisms
  list(APPEND CUDA_NVCC_FLAGS --gpu-architecture=compute_30 --gpu-code=compute_30)

  # Issue with cuda 7.5 on Ubuntu 16.04
  # https://github.com/BVLC/caffe/issues/4046
  list(APPEND CUDA_NVCC_FLAGS -D_FORCE_INLINES)

  # Allow multiple processes to use the GPU concurrently
  # https://github.com/mtazzari/galario/issues/60
  list(APPEND CUDA_NVCC_FLAGS --default-stream per-thread)

  if(GALARIO_TIMING)
    list(APPEND CUDA_NVCC_FLAGS -DGALARIO_TIMING)
  endif()

  SET(CUDA_PROPAGATE_HOST_FLAGS ON)

  # Specify include directories
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

  # copy over common file and rename to .cu, else cmake doesn't call nvcc on it
  set(common_cu "${CMAKE_CURRENT_BINARY_DIR}/cuda_lib.cu")
  configure_file(${common_cpp} ${common_cu})

  # first compile float version
  cuda_add_library(galario_single_cuda ${common_cu})

  # add_definitions(-DDOUBLE_PRECISION) only add now because undefined
  # means single precision, else we'd have to remove the -D... from
  # NVCC_FLAGS and that would be more work
  list(APPEND CUDA_NVCC_FLAGS -DDOUBLE_PRECISION)
  cuda_add_library(galario_cuda ${common_cu})

  foreach(t IN ITEMS galario_single_cuda galario_cuda)
    cuda_add_cublas_to_target(${t})
    cuda_add_cufft_to_target(${t})
    list(APPEND install_libs ${t})
  endforeach()
endif() # cuda

###
# install
###
# it would be nice if directory would let me access all libraries but there is no such property
# https://cmake.org/cmake/help/v3.0/manual/cmake-properties.7.html
install (TARGETS ${install_libs}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )
install(FILES galario.h galario_py.h galario_defs.h DESTINATION include)

###
# testing
###
add_executable(galario_test_cpp galario_test.cpp)
add_test(cpp_compile_test galario_test_cpp)

foreach(t IN ITEMS galario_test_cpp)
  target_link_libraries(${t} galario ${FFTW3_LIBRARIES})
endforeach()
