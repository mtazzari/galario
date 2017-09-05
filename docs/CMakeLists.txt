# https://eb2.co/blog/2012/03/sphinx-and-cmake-beautiful-documentation-for-c-projects/

if(NOT DEFINED SPHINX_THEME)
    set(SPHINX_THEME default)
endif()

if(NOT DEFINED SPHINX_THEME_DIR)
    set(SPHINX_THEME_DIR)
endif()

# configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

# HTML output directory
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
    "${BINARY_BUILD_DIR}/conf.py"
    @ONLY)

# sphinx requires a _static directory, even if it's empty
file(MAKE_DIRECTORY "${BINARY_BUILD_DIR}/_static")

# github pages imposes jekyll theme by default. To avoid that, we need
# a `.nojekyll` file in the gh-pages branch file(WRITE
# "${SPHINX_HTML_DIR}/.nojekyll" "")

add_custom_target(docs ALL
    ${SPHINX_EXECUTABLE}
        -q -b html
        -c "${BINARY_BUILD_DIR}"
        -d "${SPHINX_CACHE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SPHINX_HTML_DIR}"
    COMMENT "Building HTML documentation with Sphinx")

# sphinx can fail if it doesn't find the theme. This should not prevent building
# the actual source code, so docs have to be built explicitly
set_target_properties(docs PROPERTIES EXCLUDE_FROM_ALL TRUE)
