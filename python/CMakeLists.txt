###
# follow https://github.com/thewtex/cython-cmake-example
###
set(common_in_pyx "${CMAKE_CURRENT_SOURCE_DIR}/libcommon.pyx")
set_source_files_properties(${common_in_pyx} PROPERTIES CYTHON_IS_CXX TRUE)

###
# Create directories and copy files.
#
# Want to compile same pyx code for both single and double
# precision. This requires some hacking because the files and python
# modules have the same name, so they need to appear in different folders
###
set(PYGALARIO_DIR "${CMAKE_CURRENT_BINARY_DIR}/pygalario")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in"
  "${PYGALARIO_DIR}/__init__.py"
  )

set(init_module_py __init_module__.py.in)

# TODO use function to avoid copying code

###
# first compile with double precision
###
# set DOUBLE_PRECISION for cython via pxi.in
set(GALARIO_DOUBLE_PRECISION 1)
set(outdir double)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/galario_config.pxi.in"
  "${PYGALARIO_DIR}/${outdir}/galario_config.pxi"
  )
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/__init_module__.py.in"
  "${PYGALARIO_DIR}/${outdir}/__init__.py"
  )
# UseCython.cmake complains if source file is in binary_dir. We don't
# want cmake to mess with the actual source files. So the config file
# has the same name for single and double precision but it is in two
# different subdirectories in the binary_dir so parallel builds don't
# get confused. Setting the include directory, we can distinguish.
set(CYTHON_FLAGS -I "${PYGALARIO_DIR}/${outdir}")
# cython_add_module(pygalario "${common_in_pyx}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PYGALARIO_DIR}/${outdir}")
cython_add_module(libcommon "${common_in_pyx}")
target_include_directories(libcommon PUBLIC "${CMAKE_SOURCE_DIR}/src")
# set DOUBLE_PRECISION when compiling cython output and including header from `src/`
target_compile_definitions(libcommon PUBLIC DOUBLE_PRECISION)
target_link_libraries(libcommon galario)

###
# then single precision
###
set(GALARIO_DOUBLE_PRECISION 0)
set(outdir single)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/galario_config.pxi.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${outdir}/galario_config.pxi"
  )
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/__init_module__.py.in"
  "${PYGALARIO_DIR}/${outdir}/__init__.py"
  )
set(CYTHON_FLAGS -I "${CMAKE_CURRENT_BINARY_DIR}/${outdir}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PYGALARIO_DIR}/${outdir}")
cython_add_module(libcommonf "${common_in_pyx}")
# don't set DOUBLE_PRECISION
target_include_directories(libcommonf PUBLIC "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(libcommonf galariof)
# rename the library because targets need to be unique
set_target_properties(libcommonf PROPERTIES OUTPUT_NAME libcommon)

if (CUDA_FOUND)
  ###
  # cuda double precision
  ###
  set(outdir double_cuda)
  set(GALARIO_DOUBLE_PRECISION 1)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/galario_config.pxi.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${outdir}/galario_config.pxi"
    )
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/__init_module__.py.in"
    "${PYGALARIO_DIR}/${outdir}/__init__.py"
    )
  set(CYTHON_FLAGS -I "${CMAKE_CURRENT_BINARY_DIR}/${outdir}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PYGALARIO_DIR}/${outdir}")
  cython_add_module(libcommon_cuda "${common_in_pyx}")
  # don't set DOUBLE_PRECISION
  target_include_directories(libcommon_cuda PUBLIC "${CMAKE_SOURCE_DIR}/src")
  target_link_libraries(libcommon_cuda galario_cuda)
  target_compile_definitions(libcommon_cuda PUBLIC DOUBLE_PRECISION)
  # rename the library because targets need to be unique
  set_target_properties(libcommon_cuda PROPERTIES OUTPUT_NAME libcommon)

  ###
  # cuda single precision
  ###
  set(GALARIO_DOUBLE_PRECISION 0)
  set(outdir single_cuda)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/galario_config.pxi.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${outdir}/galario_config.pxi"
    )
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/__init_module__.py.in"
    "${PYGALARIO_DIR}/${outdir}/__init__.py"
    )
  set(CYTHON_FLAGS -I "${CMAKE_CURRENT_BINARY_DIR}/${outdir}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PYGALARIO_DIR}/${outdir}")
  cython_add_module(libcommonf_cuda "${common_in_pyx}")
  # don't set DOUBLE_PRECISION
  target_include_directories(libcommonf_cuda PUBLIC "${CMAKE_SOURCE_DIR}/src")
  target_link_libraries(libcommonf_cuda galariof_cuda)
  # rename the library because targets need to be unique
  set_target_properties(libcommonf_cuda PROPERTIES OUTPUT_NAME libcommon)
endif(CUDA_FOUND)

###
# tests
###
set(py_test_file test_all.py)

set(PYVFIT_STATIC_DIR /home/beaujean/workspace/protoplanetary/pyvfit/pyvfit/static CACHE FILEPATH "Static directory of pyvfit with reference data sets and outputs")

# use configure_file so it's copied on every change of the source
# http://stackoverflow.com/a/34800667/987623
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/${py_test_file}"
  "${CMAKE_CURRENT_BINARY_DIR}/"
  )
add_test(NAME CanImport COMMAND ${PYTHON_EXECUTABLE} -c "import pygalario")

# https://github.com/UCL/GreatCMakeCookOff/wiki/Adding-py.test-to-ctest
include(AddPyTest)

# I don't understand the first arg but it seems to work if py.test already installed
setup_pytest("${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/py.test.sh")

# the source name has to match test_*.py
# https://github.com/UCL/GreatCMakeCookOff/issues/56
add_pytest("${CMAKE_CURRENT_BINARY_DIR}/${py_test_file}" WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
