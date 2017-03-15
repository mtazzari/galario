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
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/double/__init__.py.in"
  "${PYGALARIO_DIR}/double/__init__.py"
  )
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/single/__init__.py.in"
  "${PYGALARIO_DIR}/single/__init__.py"
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
# then single precision
###
# set(GALARIO_DOUBLE_PRECISION 0)
# configure_file(
#   "${CMAKE_CURRENT_SOURCE_DIR}/galario_config.pxi.in"
#   "${CMAKE_CURRENT_BINARY_DIR}/single/galario_config.pxi"
#   )
# set(CYTHON_FLAGS -I "${CMAKE_CURRENT_BINARY_DIR}/single")
# cython_add_module(pygalariof "${common_in_pyx}")
# # don't set DOUBLE_PRECISION
# target_include_directories(pygalariof PUBLIC "${CMAKE_SOURCE_DIR}/src")
# target_link_libraries(pygalariof galariof)

add_test(NAME CanImport COMMAND python3 -c "import pygalario")
