# Define cython wrappers for double and cuda libs
function(wrap_lib)
  ###
  # define the arguments
  ###
  set(options DOUBLE CUDA)
  cmake_parse_arguments(WRAP_LIB "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN} )

  ###
  # define file names based on args
  ###
  if(WRAP_LIB_DOUBLE)
    set(GALARIO_DOUBLE_PRECISION 1)
    # set DOUBLE_PRECISION for cython via pxi.in
    set(outdir "${PYGALARIO_DIR}/double")
    # no suffix for double
  else()
    set(GALARIO_DOUBLE_PRECISION 0)
    set(outdir "${PYGALARIO_DIR}/single")
    # append `f` for single precision as for FFTW3
    set(suffix "_single")
  endif()
  if(WRAP_LIB_CUDA)
    set(suffix "${suffix}_cuda")
    set(outdir "${outdir}_cuda")
  endif()

  ###
  # add the targets
  ###
  # target name
  set(libcommon "libcommon${suffix}")

  # UseCython.cmake complains if source file is in binary_dir. We don't
  # want cmake to mess with the actual source files. So the config file
  # has the same name for single and double precision but it is in two
  # different subdirectories in the binary_dir so parallel builds don't
  # get confused. Setting the include directory, we can distinguish.
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/galario_config.pxi.in"
    "${outdir}/galario_config.pxi"
    )
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/__init_module__.py.in"
    "${outdir}/__init__.py"
    )
  set(CYTHON_FLAGS -I "${outdir}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${outdir}")
  cython_add_module(${libcommon} "${common_in_pyx}")
  target_include_directories(${libcommon} PUBLIC "${CMAKE_SOURCE_DIR}/src")

  # set DOUBLE_PRECISION when compiling cython output and including header from `src/`
  if (WRAP_LIB_DOUBLE)
    target_compile_definitions(${libcommon} PUBLIC DOUBLE_PRECISION)
  endif()
  # get rid of cython/numpy deprecation warning, gets rid of all other preprocessor warnings, too
  target_compile_options(${libcommon} PUBLIC "-Wno-cpp")
  target_link_libraries(${libcommon} galario${suffix})
  # rename the library because targets need to be unique
  set_target_properties(${libcommon} PROPERTIES OUTPUT_NAME libcommon)
endfunction()
