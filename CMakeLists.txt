cmake_minimum_required(VERSION 3.0)
project(galario)
set(PACKAGE_VERSION 0.1)

# we can't use static libs with python
set(BUILD_SHARED_LIBS ON)
add_subdirectory(src)
