# follow https://bloerg.net/2012/11/10/cmake-and-distutils.html
# find_program(PYTHON "python")

# if (PYTHON)
#   set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
#   set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
#   set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/galario/__init__.py")
#   set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

#   configure_file(${SETUP_PY_IN} ${SETUP_PY})

#   add_custom_command(OUTPUT  ${OUTPUT}
#     COMMAND ${PYTHON} ${SETUP_PY} build
#     COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
#     DEPENDS ${DEPS})

#   add_custom_target(py_galario ALL DEPENDS ${OUTPUT})

#   install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
# else()
#   message(WARNING "Could not find python. No bindings are generated.")
# endif()

#####################

# follow https://github.com/thewtex/cython-cmake-example
# set(common_in_pyx "${CMAKE_CURRENT_SOURCE_DIR}/libcommon.pyx")
# set(common_double_pyx "${CMAKE_CURRENT_BINARY_DIR}/libcommon_double.pyx")
# set(common_single_pyx "${CMAKE_CURRENT_BINARY_DIR}/libcommon_single.pyx")

# # TODO create a macro to avoid repeating myself?

# # first compile with double precision
# set(DOUBLE_PRECISION True)
# configure_file("${common_in_pyx}" "${common_double_pyx}")
# # compile as C++
# set_source_files_properties(${common_double_pyx} PROPERTIES CYTHON_IS_CXX TRUE)
# # it seems it doesn't work if  source (configure) file is in binary directory
# #
# # CMake Error at cmake/UseCython.cmake:111 (get_directory_property):
# #   get_directory_property DIRECTORY argument provided but requested directory
# #   not found.  This could be because the directory argument was invalid or, it
# #   is valid but has not been processed yet.
# # Call Stack (most recent call first):
# #   cmake/UseCython.cmake:274 (compile_pyx)
# #   python/CMakeLists.txt:39 (cython_add_module)
# cython_add_module(pygalario "${common_double_pyx}")
# target_include_directories(pygalario PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# set(DOUBLE_PRECISION False)
# configure_file("${common_in_pyx}" "${common_single_pyx}")
# # compile as C++
# set_source_files_properties(${common_single_pyx} PROPERTIES CYTHON_IS_CXX TRUE)
# # cython_add_module(pygalariof "${common_single_pyx}")

#####################################3

set(common_in_pyx "${CMAKE_CURRENT_SOURCE_DIR}/libcommon.pyx")
# first compile with double precision

# set(GALARIO_DOUBLE_PRECISION 1)
# configure_file(
#   "${CMAKE_CURRENT_SOURCE_DIR}/galario_config.hpp.in"
#   "${CMAKE_CURRENT_BINARY_DIR}/double/galario_config.hpp"
#   )
set_source_files_properties(${common_in_pyx} PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(pygalario "${common_in_pyx}")
target_include_directories(pygalario PUBLIC
  "${CMAKE_CURRENT_BINARY_DIR}/double"
  "${CMAKE_SOURCE_DIR}/src")
# needed when including code from `src/`
target_compile_definitions(pygalario PUBLIC DOUBLE_PRECISION)
# include_directories("${CMAKE_CURRENT_BINARY_DIR}/double")

# set(GALARIO_DOUBLE_PRECISION 0)
# configure_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/galario_config.hpp.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/single/galario_config.hpp"
#     )
