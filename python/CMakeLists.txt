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

###
# first compile with double precision
###
# set DOUBLE_PRECISION for cython via pxi.in
set(GALARIO_DOUBLE_PRECISION 1)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/galario_config.pxi.in"
  "${PYGALARIO_DIR}/double/galario_config.pxi"
  )
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/double/__init__.py.in"
  "${PYGALARIO_DIR}/double/__init__.py"
  )
# UseCython.cmake complains if source file is in binary_dir. We don't
# want cmake to mess with the actual source files. So the config file
# has the same name for single and double precision but it is in two
# different subdirectories in the binary_dir so parallel builds don't
# get confused. Setting the include directory, we can distinguish.
set(CYTHON_FLAGS -I "${PYGALARIO_DIR}/double")
# cython_add_module(pygalario "${common_in_pyx}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PYGALARIO_DIR}/double")
cython_add_module(libcommon "${common_in_pyx}")
target_include_directories(libcommon PUBLIC "${CMAKE_SOURCE_DIR}/src")
# set DOUBLE_PRECISION when compiling cython output and including header from `src/`
target_compile_definitions(libcommon PUBLIC DOUBLE_PRECISION)
target_link_libraries(libcommon galario)

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
  "${CMAKE_CURRENT_SOURCE_DIR}/${outdir}/__init__.py.in"
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
# then single precision
###
set(GALARIO_DOUBLE_PRECISION 0)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/galario_config.pxi.in"
  "${CMAKE_CURRENT_BINARY_DIR}/single/galario_config.pxi"
  )
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/single/__init__.py.in"
  "${PYGALARIO_DIR}/single/__init__.py"
  )
set(CYTHON_FLAGS -I "${CMAKE_CURRENT_BINARY_DIR}/single")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PYGALARIO_DIR}/single")
cython_add_module(libcommonf "${common_in_pyx}")
# don't set DOUBLE_PRECISION
target_include_directories(libcommonf PUBLIC "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(libcommonf galariof)
# rename the library because targets need to be unique
set_target_properties(libcommonf PROPERTIES OUTPUT_NAME libcommon)

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
  "${CMAKE_CURRENT_SOURCE_DIR}/${outdir}/__init__.py.in"
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

###
# tests
###
add_test(NAME CanImport COMMAND python3 -c "import pygalario")
