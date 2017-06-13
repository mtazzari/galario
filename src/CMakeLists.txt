# C++ 11 support http://stackoverflow.com/questions/37621342/ for cmake >= 2.8
# these options are understood by both gcc and nvcc
set(compile_options -std=c++11 -O3)
# these need to be wrapped in -Xcompiler for nvcc
set(extra_compile_options  -Wall -fno-strict-aliasing -pthread -fwrapv)
# we set CXX_FLAGS because they are then automatically propagated also to nvcc with -Xcompiler option
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${compile_options}")
add_compile_options(${compile_options} ${extra_compile_options})

# only affects CPU compilation
# add_compile_options("${compile_options}" -pthread -fwrapv)

# we compile the same source file multiple times, varying preprocessor definitions
set(common_cpp "${CMAKE_CURRENT_SOURCE_DIR}/galario.cpp")

# compile single and double precision in separate libraries
add_library(galario_single ${common_cpp})

add_library(galario ${common_cpp})
target_compile_definitions(galario PUBLIC DOUBLE_PRECISION)

set(install_libs galario_single galario)

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  # https://cmake.org/Bug/view.php?id=9075
  # http://stackoverflow.com/questions/24532853/how-to-add-linker-flag-for-libraries-with-cmake
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(FFTW_PARALLEL "OPENMP")
endif()

# we need fftw with (single+double)x(serial+openmp)
find_package(FFTW3 COMPONENTS SINGLE DOUBLE ${FFTW_PARALLEL})

foreach(t IN ITEMS galario_single galario)
  target_link_libraries(${t} ${FFTW3_LIBRARIES})
endforeach()

if (CUDA_FOUND)
# if (0)

  # TODO I couldn't get nvcc to work with flags propagated and I still
  # need to pass every arg in ${extra_compile_options} to
  # nvcc. Probably a function will do it that emits "-Xcompiler arg1 -Xcompiler arg2 ..."
  list(APPEND CUDA_NVCC_FLAGS ${compile_options})

  # for timing test, we want to avoid the start-up time from JIT, so we create fat binaries
  # http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list
  list(APPEND CUDA_NVCC_FLAGS --gpu-architecture=compute_30 --gpu-code=compute_30,sm_53)

  # Issue with cuda 7.5 on Ubuntu 16.04
  # https://github.com/BVLC/caffe/issues/4046
  list(APPEND CUDA_NVCC_FLAGS -D_FORCE_INLINES)

  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

  # Specify include directories
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

  # Specify library paths
  link_directories(
    # /opt/foobar/lib
    # /opt/joestuff/lib
    )

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

  # ignored
  # target_compile_definitions(galario_cuda PUBLIC DOUBLE_PRECISION)
endif()


# it would be nice if directory would let me access all libraries but there is no such property
# https://cmake.org/cmake/help/v3.0/manual/cmake-properties.7.html
install (TARGETS ${install_libs}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )
install(FILES galario.hpp DESTINATION include)
