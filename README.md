galario
=======
Gpu Accelerated Library for Analysing Radio Interferometry Observations

build instructions
------------------

With the default configuration

    git clone https://github.com/mtazzari/galario.git
    cd galario
    mkdir build && cd build
    cmake .. && make

Before playing with the options, it's best to remove the cache

    rm build/CMakeCache.txt

### C compiler

    cmake -DCMAKE_C_COMPILER=$GCC_BASE/bin/gcc ..

### python

Specify a python version. This is useful if python 2.7 and 3.x are in
the system and conflicting versions of the interpreter and the
libraries are found. In `build/`, do

    cmake -DPython_ADDITIONAL_VERSIONS=3.5 ..

galario should work with both python 2 and 3. To create conda environments

    conda create --name galario2 python=2 numpy cython astropy pytest
    conda create --name galario3 python=3 numpy cython astropy pytest

cmake may get confused with the conda python and the system
python. This is a general problem
https://cmake.org/Bug/view.php?id=14809

A workaround to help cmake find the interpreter and the libs from the
currently loaded conda environment is

    cmake -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} ..

### FFTW

The FFTW libraries are required for the CPU version of galario.
To install FFTW follow the instructions at http://www.fftw.org.
galario requires the following FFTW libraries:

* libfftw3              # double precision
* libfftw3f             # single precision
* libfftw3_omp          # double precision with OpenMP
* libfftw3f_omp         # single precision with OpenMP

galario has been tested with FFTW 3.3.6.

To compile FFTW on a Mac download the .tar.gz from FFTW website you have to explicitly
enable the build of dynamic (shared) library with --enable-shared option, and run multiple times
./configure && make && make install in order to create the libraries listed above:

    cd fftw-<version>/
    mkdir d_p && cd d_p && \
      CC=/usr/local/bin/gcc ../configure --enable-shared && make && sudo make install && cd ..
    mkdir s_p && cd s_p && \
      CC=/usr/local/bin/gcc ../configure --enable-shared --enable-single && make && sudo make install && cd ..
    mkdir d_p_omp && cd d_p_omp && \
      CC=/usr/local/bin/gcc ../configure --enable-shared --enable-openmp && make && sudo make install && cd ..
    mkdir s_p_omp && cd s_p_omp && \
      CC=/usr/local/bin/gcc ../configure --enable-shared --enable-single --enable-openmp && make && sudo make install && cd ..

If you have no sudo rights to install FFTW libraries, then provide a directory via `make install --prefix="/path/to/fftw"`.
Before building galario, `FFTW_HOME` has to be set equal to the installation directory of FFTW, e.g. `FFTW_HOME="/usr/local/lib/"`
in the default case, or to the prefix specified during the fftw installation.

To speedup building FFTW, you may add the -jN flag to the make commands above, e.g. `make -jN`, where N is an integer
equal to the number of cores you want to use. E.g., on a 4-cores machine, you can do `make -j4`. To use -j4 as default, you can
create an alias with

    alias make="make -j4"

To find FFTW3 in a nonstandard directory, say `$FFTW_HOME`, tell `cmake`
about it

    cmake -DCMAKE_PREFIX_PATH=${FFTW_HOME} ..

For multiple directories, use a `;` between directories

    cmake -DCMAKE_PREFIX_PATH=${FFTW_HOME};/opt/something/else ..

In case the directory with the header files is not inferred correctly,

    cmake -DCMAKE_CXX_FLAGS="-I${FFTW_HOME}/include" ..

### cuda

`cmake` tests for compilation on the gpu with cuda by default except on the mac
where version conflicts between the nvidia compiler and the C++ compiler often lead to problems; see [https://github.com/mtazzari/galario/issues/30](issue #30).

To manually turn off cuda support, use

    cmake -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=0 ..

To force searching for cuda, for example on the mac, do

    cmake -DGALARIO_FORCE_CUDA=1 ..

installation
------------

To specify where to install, do the conventional

    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/lib ..

and after building run

    make install

Note that by default the C libraries and the python bindings are installed under
the same prefix. If you want to install the python bindings elsewhere, there is
an extra cache variable `GALARIO_PYTHON_PKG_DIR` that you can edit with `ccmake
.` after running `cmake`. An active conda environment is used to initialize
`GALARIO_PYTHON_PKG_DIR`. For example,

    conda activate myenv
    cmake -DCMAKE_INSTALL_PREFIX=/some/prefix ..
    make && make install

will output the following in the install step

    -- Installing: /some/prefix/lib/libgalario.so
    -- Installing: /path/to/conda/envs/myenv/lib/python2.7/site-packages/galario/single/__init__.py

testing
-------

After building, just run `ctest -V --output-on-failure` in `build/`.

Every time `python/test_galario.py` is modified, it has to be copied over to the
build directory: only when run there, `import pygalario` works. The copy is
performed in the configure step, `cmake` detects changes so always run `make` first.

`py.test` fails if it cannot collect any tests. This can be caused by C errors.
To debug the testing, first find out the exact command of the test

    make && ctest -V

`py.test` captures the output from the test, in particular from C to stderr.
Force it to show all output

    make && python/py.test.sh -sv python_package/tests/test_galario.py

By default, tests run on the GPU if code is available in `galario`. Deactivate
by calling `... py.test.sh --gpu=0 ...`. To select the parametrized test
`test_sample`, `... py.test.sh -k sample`.

A cuda error such as

    [ERROR] Cuda call /home/beaujean/workspace/protoplanetary/galario/build2/src/cuda_lib.cu: 815
    invalid argument

can mean that code cannot be executed on the GPU at all rather than that
specific call being invalid. Check if `nvidia-smi` runs

    $ nvidia-smi
    Failed to initialize NVML: Driver/library version mismatch

documentation
-------------

    make docs

creates output in `docs/html` under the build directory. Add content to
`docs/index.rst` or the files linked to therein. The `docs` are not build by
default, only upon request.

Within a conda environment, `conda install sphinx` to have a `sphinx` version
that matches the python version. As the `galario` library needs to be imported
when building the docs, the import would fail otherwise. Remove the
`CMakeCache.txt` and rerun `cmake`, and observe which location of `sphinx` is reported, for example

    -- Found Sphinx: /home/myuser/.local/miniconda3/envs/galario3/bin/sphinx-build
