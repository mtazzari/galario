galario
=======
Gpu Accelerated Library for Analysing Radio Interferometry Observations

build instructions
------------------

With the default configuration

    git clone ...
    cd galario
    mkdir build && cd build
    cmake .. && make

Before changing the basics, it's best to remove the cache

    rm build/CMakeCache.txt

Specify a python version. This is useful if python 2.7 and 3.x are in
the system and conflicting versions of the interpreter and the
libraries are found. In `build/`, do

    cmake -DPython_ADDITIONAL_VERSIONS=3.5 ..

python environment
------------------

galario should work with both python 2 and 3

    conda create --name galario2 python=2 numpy cython
    conda create --name galario3 python=3 numpy cython

If you want to run the unit test, you need some more packages

    conda create --name galario2 python=2 numpy cython astropy pytest

cmake may get confused with the conda python and the system
python. This is a general problem
https://cmake.org/Bug/view.php?id=14809

A workaround to help cmake find the interpreter and the libs from the
conda environment is

    cmake -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} ..

testing
-------

After building, just run `ctest -V --output-on-failure` in `build/`.

To compare with pyvfit need `python2.7` and `pyvfit` installed in
developer mode so we can pick up the `static` directory from the
location of `pyvfit` itself.

My current one liner to get going is

    cmake \
        -DGreatCMakeCookOff_DIR=$HOME/workspace/GreatCMakeCookOff/cmake \
        -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
        .. \
    && make all test

Every time `python/test_all.py` is modified, it has to be copied over
to the build directory: only when run there, `import pygalario`
works. The copy is performed in the build step but I couldn't get the
dependency injected, so to run the tests, you have to do `make && make
test` or `make && ctest`.

installation
------------

Do the conventional

    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/lib

Note that by default the C libraries and the python bindings are
installed under the same prefix. If you want to install python
elsewhere, there is an extra cache variable `GALARIO_PYTHON_PKG_DIR`
that you can edit with `ccmake .` after running `cmake`. If conda is
used, we give it higher precedence. For example,

    conda activate myenv
    cmake -DCMAKE_INSTALL_PREFIX=/some/prefix ..
    make && make install

will output the following in the install step

    -- Installing: /some/prefix/lib/libgalario.so
    -- Installing: /path/to/conda/envs/myenv/lib/python2.7/site-packages/galario/single/__init__.py
