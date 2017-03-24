# TODO nicer cmake stuff in https://github.com/UCL/GreatCMakeCookOff/tree/master/modules
find_package(FFTW REQUIRED)

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
set(common_cpp "${CMAKE_CURRENT_SOURCE_DIR}/cuda_lib.cpp")

# compile single and double precision in separate libraries
add_library(galariof ${common_cpp})

add_library(galario ${common_cpp})
target_compile_definitions(galario PUBLIC DOUBLE_PRECISION)

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  # https://cmake.org/Bug/view.php?id=9075
  # http://stackoverflow.com/questions/24532853/how-to-add-linker-flag-for-libraries-with-cmake
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# TODO doesn't include openmp or mpi versions
# we need (single+double)x(serial+threads)
set(FFTW_LIBRARIES ${FFTW_LIB} ${FFTWF_LIB} ${FFTW_THREADS_LIB} ${FFTWF_THREADS_LIB})
target_link_libraries(galario "${FFTW_LIBRARIES}")
target_link_libraries(galariof "${FFTW_LIBRARIES}")

if (CUDA_FOUND)
# if (0)

  # TODO I couldn't get nvcc to work with flags propagated and I still
  # need to pass every arg in ${extra_compile_options} to
  # nvcc. Probably a function will do it that emits "-Xcompiler arg1 -Xcompiler arg2 ..."
  list(APPEND CUDA_NVCC_FLAGS ${compile_options} -Wno-deprecated-gpu-targets)

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
  cuda_add_library(galariof_cuda ${common_cu})

  # add_definitions(-DDOUBLE_PRECISION) only add now because undefined
  # means single precision, else we'd have to remove the -D... from
  # NVCC_FLAGS and that would be more work
  list(APPEND CUDA_NVCC_FLAGS -DDOUBLE_PRECISION)
  cuda_add_library(galario_cuda ${common_cu})

  # ignored
  # target_compile_definitions(galario_cuda PUBLIC DOUBLE_PRECISION)

  # For linking ...
  # Specify target & libraries to link it with
  target_link_libraries(
    galario_cuda
    # -lfoobar
    # -ljoestuff
    )
endif()
