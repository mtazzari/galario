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

include(wrap_lib.cmake)
wrap_lib()
wrap_lib(DOUBLE)

if (CUDA_FOUND)
  wrap_lib(CUDA)
  wrap_lib(DOUBLE CUDA)
endif()

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
