###############################################################################
# This file is part of GALARIO:                                               #
# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
#                                                                             #
# Copyright (C) 2017-2020, Marco Tazzari, Frederik Beaujean, Leonardo Testi.  #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the Lesser GNU General Public License as published by #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                        #
#                                                                             #
# For more details see the LICENSE file.                                      #
# For documentation see https://mtazzari.github.io/galario/                   #
###############################################################################

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
set(PYGALARIO_DIR "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}")
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
# installation
###
# to set ${PYTHON_PKG_DIR}. It is prepended by ${CMAKE_INSTALL_PREFIX} if set
include(PythonInstall)

# I guess variable has to be in cache to survive reinvocations of cmake that happen during building
set(GALARIO_PYTHON_PKG_DIR "${PYTHON_PKG_DIR}" CACHE PATH "Current python install path")

# on unix: install .py and .so relative to ${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT}
install(DIRECTORY ${PYGALARIO_DIR}
  DESTINATION ${SKBUILD_PLATLIB_DIR}
  FILES_MATCHING REGEX ".*py$|.*so$"
  )

###
# tests
###
configure_file(speed_benchmark.py "${CMAKE_CURRENT_BINARY_DIR}" COPYONLY)
configure_file(utils.py "${CMAKE_CURRENT_BINARY_DIR}" COPYONLY)
configure_file(speed_baseline.sh "${CMAKE_CURRENT_BINARY_DIR}" COPYONLY)

# TODO avoid installation of test scripts
# http://stackoverflow.com/questions/13381742/cmake-exclude-executable-during-install
set(galario_test test_galario.py)

# use configure_file so it's copied on every change of the source
# http://stackoverflow.com/a/34800667/987623
configure_file("${galario_test}" "${CMAKE_CURRENT_BINARY_DIR}" COPYONLY)
add_test(NAME CanImport COMMAND ${PYTHON_EXECUTABLE} -c "import ${PROJECT_NAME}")

# https://github.com/UCL/GreatCMakeCookOff/wiki/Adding-py.test-to-ctest
include(AddPyTest)

# I don't understand the first arg but it seems to work if py.test already installed
setup_pytest("${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/py.test.sh")

# the source name has to match test_*.py
# https://github.com/UCL/GreatCMakeCookOff/issues/56
#
# need to run in binary_dir so import of pygalario works
add_pytest("${CMAKE_CURRENT_BINARY_DIR}/${galario_test}" NOINSTALL CMDLINE --cov=./)
