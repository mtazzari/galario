==================================
How to build and install `galario`
==================================

System Requirements
-------------------
To compile `galario` you will need:

 * the `FFTW libraries <http://www.fftw.org>`_, for the CPU version: more details are given :ref:`below <fftw_requirement>`.
 * the `CUDA toolkit <https://developer.nvidia.com/cuda-toolkit>`_ later than 8.0, for the GPU version: it can be easily installed from the `NVIDIA website <https://developer.nvidia.com/cuda-toolkit>`_.
 * the GNU `gcc` and `g++` compilers, for CPU and GPU versions.

.. warning::
    On Mac OS, the GNU compilers must be manually downloaded and installed, e.g. following `these instructions <http://hpc.sourceforge.net>`_.
    The default `gcc`/`g++` compilers shipped with the OS are aliases for the `clang` compiler, which does does not currently support `openMP`.

Quick steps to build and install
--------------------------------
Here a quick summary to compile and install `galario`, :ref:`below <detailed_build_instructions>`
more detailed instructions in case you need.

The following procedure will always compile and install the CPU version of `galario`.
On a system with a CUDA-enabled GPU card, also the GPU version will be compiled and installed.

 1. clone the repository and create a directory where to build `galario`:

    .. code-block:: bash

        git clone https://github.com/mtazzari/galario.git
        cd galario
        mkdir build && cd build

 2. to make the compilation easier, let's work in a Python environment. `galario` works with both Python 2 and 3.
    If you are using anaconda Python you can create the environment with:

    .. code-block:: bash

        conda create --name galario3 python=3 numpy cython pytest

 3. use `cmake` to prepare the compilation and `make all` to compile. From within `galario/build/`:

    .. code-block:: bash

        CC="/path/to/gcc" CXX="/path/to/g++" cmake ../ && make all

    where typically CC="/usr/local/bin/gcc" and CXX="/usr/local/bin/g++" but may vary depending on the system.

    This command will produce configuration and compilation logs listing all the libraries and the compilers that are being used.

These instructions should be sufficient in most cases, but if you have problems or want more fine-grained control,
check out the details below. If you find issues or are stuck in one of these steps, consider writing us an email
or opening an issue on the `GitHub <https://github.com/mtazzari/galario.git>`_ repository.


.. _detailed_build_instructions:

build instructions
------------------

With the default configuration

.. code-block:: bash

    git clone https://github.com/mtazzari/galario.git
    cd galario
    mkdir build && cd build
    cmake .. && make

Before playing with the `cmake` options, it's best to remove the cache

.. code-block:: bash

    rm build/CMakeCache.txt

### C++ compiler

.. code-block:: bash

    cmake -DCMAKE_CXX_COMPILER=$GCC_BASE/bin/g++ ..

### optimization

See

.. code-block:: bash

    cmake --help-variable CMAKE_BUILD_TYPE

The default is `Release`. If you want debug symbols as well, use
`RelWithDebInfo`. To turn off optimization

.. code-block:: bash

    cmake -DCMAKE_BUILD_TYPE=Debug

To turn on even more aggressive optimization, pass the flags
directly. For example for gcc

.. code-block:: bash

    cmake -DCMAKE_CXX_FLAGS='-march=native -ffast-math'



### python

Specify a python version. This is useful if python 2.7 and 3.x are in
the system and conflicting versions of the interpreter and the
libraries are found. In `build/`, do

.. code-block:: bash

    cmake -DPython_ADDITIONAL_VERSIONS=3.5 ..

galario should work with both python 2 and 3. To create conda environments

.. code-block:: bash

    conda create --name galario2 python=2 numpy cython pytest
    conda create --name galario3 python=3 numpy cython pytest

To run the tests, install some more dependencies within the environment

.. code-block:: bash

    conda config --add channels conda-forge
    conda install pyfftw scipy

cmake may get confused with the conda python and the system
python. This is a general problem
https://cmake.org/Bug/view.php?id=14809

A workaround to help cmake find the interpreter and the libs from the
currently loaded conda environment is

.. code-block:: bash

    cmake -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} ..

If you still have problems, after the `cmake` command, check whether the FFTW libraries with openMP flags are found and
whether the path to Python is correctly set to the path of the conda environment in use, e.g. in this example `/home/user/anaconda/envs/galario3`.

.. _fftw_requirement:

### FFTW

The FFTW libraries are required for the CPU version of galario.
You can check if they are installed on your system by checking if **all** libraries listed below are
present in `/usr/local/lib/`.
To install FFTW follow the instructions at http://www.fftw.org.
galario requires the following FFTW libraries:

* libfftw3              # double precision
* libfftw3f             # single precision
* libfftw3_omp          # double precision with OpenMP
* libfftw3f_omp         # single precision with OpenMP

galario has been tested with FFTW 3.3.6.

On a Mac
~~~~~~~~
To compile FFTW on a Mac download the .tar.gz from FFTW website you have to explicitly
enable the build of dynamic (shared) library with --enable-shared option, and run multiple times
./configure && make && make install in order to create the libraries listed above:

.. code-block:: bash

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

.. code-block:: bash

    alias make="make -j4"

To find FFTW3 in a nonstandard directory, say `$FFTW_HOME`, tell `cmake`
about it

.. code-block:: bash

    cmake -DCMAKE_PREFIX_PATH=${FFTW_HOME} ..

For multiple directories, use a `;` between directories

.. code-block:: bash

    cmake -DCMAKE_PREFIX_PATH=${FFTW_HOME};/opt/something/else ..

In case the directory with the header files is not inferred correctly,

.. code-block:: bash

    cmake -DCMAKE_CXX_FLAGS="-I${FFTW_HOME}/include" ..

In case the openmp libraries are not in `${FFTW_HOME}/lib`

.. code-block:: bash

    cmake -DCMAKE_LIBRARY_PATH="${FFTW_OPENMP_LIBDIR}" ..

### cuda

`cmake` tests for compilation on the gpu with cuda by default except on the mac
where version conflicts between the nvidia compiler and the C++ compiler often lead to problems; see [https://github.com/mtazzari/galario/issues/30](issue #30).

To manually turn off cuda support, use

.. code-block:: bash

    cmake -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=1 ..

To force searching for cuda, for example on the mac, do

.. code-block:: bash

    cmake -DGALARIO_FORCE_CUDA=1 ..

### timing

For testing purposes, the time in seconds taken by selected functions called from `galario_sample` is printed to `stdout`. This features is off by default and activated by

.. code-block:: bash

    cmake -DGALARIO_TIMING=1 ..

installation
------------

To specify where to install, do the conventional

.. code-block:: bash

    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/lib ..

and after building run

.. code-block:: bash

    make install

Note that by default the C libraries and the python bindings are installed under
the same prefix. If you want to install the python bindings elsewhere, there is
an extra cache variable `GALARIO_PYTHON_PKG_DIR` that you can edit with `ccmake
.` after running `cmake`. An active conda environment is used to initialize
`GALARIO_PYTHON_PKG_DIR`. For example,

.. code-block:: bash

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

.. code-block:: bash

    make && ctest -V

`py.test` captures the output from the test, in particular from C to stderr.
Force it to show all output

.. code-block:: bash

    make && python/py.test.sh -sv python_package/tests/test_galario.py

By default, tests do not run on the GPU. Activate by calling
`... py.test.sh --gpu=1 ...`. To select the parametrized test
`test_sample`, `... py.test.sh -k sample`.

A cuda error such as

.. code-block:: bash

    [ERROR] Cuda call /home/beaujean/workspace/protoplanetary/galario/build2/src/cuda_lib.cu: 815
    invalid argument

can mean that code cannot be executed on the GPU at all rather than that
specific call being invalid. Check if `nvidia-smi` runs

.. code-block:: bash

    $ nvidia-smi
    Failed to initialize NVML: Driver/library version mismatch

documentation
-------------
.. code-block:: bash

    make docs

creates output in `docs/html` under the build directory. Add content to
`docs/index.rst` or the files linked to therein. The `docs` are not build by
default, only upon request.

Within a conda environment, `conda install sphinx` to have a `sphinx` version
that matches the python version. As the `galario` library needs to be imported
when building the docs, the import would fail otherwise. Remove the
`CMakeCache.txt` and rerun `cmake`, and observe which location of `sphinx` is reported, for example

    -- Found Sphinx: /home/myuser/.local/miniconda3/envs/galario3/bin/sphinx-build
