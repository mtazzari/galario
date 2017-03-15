find_package(FFTW REQUIRED)

# C++ 11 support http://stackoverflow.com/questions/37621342/
add_compile_options(-std=c++11) # CMake 2.8.12 or newer
add_compile_options(-pthread -fwrapv -Wall -fno-strict-aliasing)

# compile single and double precision in separate libraries
add_library(galario cuda_lib.cpp)
target_compile_definitions(galario PUBLIC DOUBLE_PRECISION)
add_library(galariof cuda_lib.cpp)

# TODO do we need to adjust linking flags as well? Probably not
# because we only create the library.
find_package(OpenMP)
if(OPENMP_FOUND)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (FFTW_FOUND)
  # TODO doesn't include openmp or mpi versions
  # we need (single+double)x(serial+threads)
  set(FFTW_LIBRARIES ${FFTW_LIB} ${FFTWF_LIB} ${FFTW_THREADS_LIB} ${FFTWF_THREADS_LIB})
  target_link_libraries(galario "${FFTW_LIBRARIES}")
  target_link_libraries(galariof "${FFTW_LIBRARIES}")
endif()
