==================================
How to build and install |galario|
==================================


Operating system
-------------------
|galario| runs on Linux and Mac OS X. Windows is not supported.

Installing via conda
--------------------

By far the easiest way to install |galario| is via `conda <https://conda.io>`_.
If you are new to `conda`, you may want to start with the minimal `miniconda
<https://repo.continuum.io/miniconda/>`_. With `conda` all dependencies are
installed automatically and you get access to |galario|'s C/C++ and python
bindings, both with support for multithreading.

.. code-block:: bash

   conda config --add channels conda-forge
   conda install galario

Due to technical limitations, the conda package does not support GPUs at the
moment. If you want to use a GPU, read on as you have to build |galario| by hand.

Build requirements
------------------

To compile |galario| you will need:

* a working internet connection (to download 1.5 MB of an external library)
* a C and C++ compiler such as `gcc` or `clang`. To use multiple threads, the compiler has to support `openMP <http://www.openmp.org/resources/openmp-compilers/>`_
* `cmake <https://cmake.org>`_ and `make`
* the `FFTW libraries <http://www.fftw.org>`_, for the CPU version: more details are given :ref:`below <fftw_requirement>`
* [optional] the `CUDA toolkit <https://developer.nvidia.com/cuda-toolkit>`_ >=8.0 for the GPU version: it can be easily installed from the `NVIDIA website <https://developer.nvidia.com/cuda-toolkit>`_
* [optional] Python and numpy for Python bindings to the CPU and GPU

.. warning::
    If you want to use the GNU compilers on Mac OS, you need to manually download and install them, e.g. following `these instructions <http://hpc.sourceforge.net>`_.
    The default `gcc`/`g++` commands shipped with the OS are aliases for the `clang` compiler that supports openMP only as of version 3.7 but unfortunately Apple usually ships an older version of `clang`.

Quick steps to build and install
--------------------------------
Here a quick summary to compile and install |galario| with default options, :ref:`below <detailed_build_instructions>` are
more detailed instructions to fine-tune the build process.

The following procedure will always compile and install the CPU version of |galario|.
On a system with a CUDA-enabled GPU card, also the GPU version will be compiled and installed.
To manually turn ON/OFF the GPU CUDA compilation, see :ref:`these instructions <build_details_cuda>` below.

 1. Clone the repository and create a directory where to build |galario|:

    .. code-block:: bash

        git clone https://github.com/mtazzari/galario.git
        cd galario
        mkdir build && cd build

 2. to make the compilation easier, let's work in a Python environment. |galario| works with both Python 2 and 3.

    For example, if you are using the `Anaconda <https://www.continuum.io/downloads>`_ distribution, you can create and
    activate a Python 3 environment with:

    .. code-block:: bash

        conda create --name galario3 python=3 numpy cython pytest
        source activate galario3

 2. Use `cmake` to prepare the compilation and `make all` to compile. From within `galario/build/`:

    .. code-block:: bash

       cmake ..

    This command will produce configuration and compilation logs listing all the libraries and the compilers that are being used.
    It will use the internet connection to automatically download `this <https://github.com/UCL/GreatCMakeCookOff>`_ additional library (1.5 MB).


 3. Use `make` to build |galario| and `make install` to install it inside the active environment:

    .. code-block:: bash

        make && make install

    If the installation fails due to permission problems, you either have to use `sudo make install`, or see the :ref:`instructions below <install_details>` to specify an alternate installation path.

..        CC="/path/to/gcc" CXX="/path/to/g++" cmake -DCMAKE_PREFIX_PATH="${FFTW_HOME};${CONDA_PREFIX}" ../ && make all
       ..
          where typically CC="/usr/local/bin/gcc" and CXX="/usr/local/bin/g++" but may be different on your system.
          `FFT_HOME` should contain the path to the FFTW libraries installed on your system and
          `CONDA_PREFIX` is automatically set to the conda environment `/anaconda/envs/galario3`.


These instructions should be sufficient in most cases, but if you have problems
or want more fine-grained control, check out the details below. If you find
issues or are stuck in one of these steps, consider writing us an email or
opening an issue on `GitHub <https://github.com/mtazzari/galario/issues>`_.

.. note::

    If you compile |galario| only for the CPU, gcc/g++ >= 4.0 works fine. If you
    compile also the GPU version, check in the |NVIDIA_docs| which gcc/g++
    versions are compatible with the `nvcc` compiler shipped with your CUDA
    Toolkit.

.. _detailed_build_instructions:

Detailed build instructions
---------------------------

The default configuration to build |galario| is

.. code-block:: bash

    git clone https://github.com/mtazzari/galario.git
    cd galario
    mkdir build && cd build
    cmake .. && make

There are many options to affect the build when `cmake` is invoked. When playing
 with options, it's best to remove the `cmake` cache first

.. code-block:: bash

    rm build/CMakeCache.txt

In the following, we assume `cmake` is invoked from the `build` directory.

Compiler
~~~~~~~~
Set the C and C++ compiler

.. code-block:: bash

   export CC="/path/to/bin/gcc"
   export CXX="/path/to/bin/g++"
   cmake ..

   # alternative
   cmake -DCMAKE_C_COMPILER=/path/to/gcc -DCMAKE_CXX_COMPILER=/path/to/g++ ..

Optimization level
~~~~~~~~~~~~~~~~~~

By default |galario| is built with all the optimizations ON. You can check this with:

.. code-block:: bash

    cmake --help-variable CMAKE_BUILD_TYPE

The default built type is `Release`, which is the fastest. If you want debug symbols as well, use `RelWithDebInfo`.

To turn on even more aggressive optimization, pass the flags directly. For example for g++:

.. code-block:: bash

    cmake -DCMAKE_CXX_FLAGS='-march=native -ffast-math' ..

Note that these further optimization might not work on any system.

To turn off optimizations:

.. code-block:: bash

    cmake -DCMAKE_BUILD_TYPE=Debug ..

.. _python_requirement:

Python
~~~~~~

To build the python bindings, we require python 2.7 or 3.x, `numpy`,
`cython`, and `pytest`. To run the tests, we additionally need
`scipy>0.14`.

Specify a Python version if Python 2.7 and 3.x are in the system and
conflicting versions of the interpreter and the libraries are found
and reported by `cmake`. In `build/`, do

.. code-block:: bash

    cmake -DPython_ADDITIONAL_VERSIONS=3.5 ..

galario should work with both python 2 and 3. For example, if you are using the `Anaconda <https://www.continuum.io/downloads>`_ distribution, you can create conda environments with

.. code-block:: bash

    # python 2
    conda create --name galario2 python=2 numpy cython pytest
    source activate galario2

    # or python3
    conda create --name galario3 python=3 numpy cython pytest
    source activate galario3

To run the tests, install some more dependencies within the environment

.. code-block:: bash

    conda install scipy

cmake may get confused with the conda python and the system
python. This is a general problem
https://cmake.org/Bug/view.php?id=14809

A workaround to help cmake find the interpreter and the libs from the
currently loaded conda environment is

.. code-block:: bash

    cmake -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} ..

If you still have problems, after the `cmake` command, check whether the FFTW
libraries with openMP flags are found and whether the path to Python is
correctly set to the path of the conda environment in use, e.g.
`/home/user/anaconda/envs/galario3`.

.. _fftw_requirement:

FFTW
~~~~

The FFTW libraries are required for the CPU version of galario.
You can check if they are installed on your system by checking if **all** libraries listed below are
present, for example in `/usr/lib` or `/usr/local/lib/`.

galario requires the following FFTW libraries:

* `libfftw3`: double precision
* `libfftw3f`: single precision
* `libfftw3_threads`: double precision with pthreads
* `libfftw3f_threads`: single precision with pthreads

galario has been tested with FFTW 3.3.6.

The easiest way to install FFTW is to use a package manager, for example `apt`
on Debian/Ubuntu or `homebrew` on the Mac. For example,

.. code-block:: bash

   sudo apt-get install libfftw3-3 libfftw3-dev


If you really want to build FFTW from
source, for example because you don't have admin rights, read on.

Manual compilation
^^^^^^^^^^^^^^^^^^

To compile FFTW, download the .tar.gz from FFTW website. On Mac OS, you have to explicitly
enable the build of dynamic (shared) library with the `--enable-shared` option, while on Linux this `should` be the default.
You can create the libraries listed above with the following lines:

.. code-block:: bash

    cd fftw-<version>/
    mkdir d_p && cd d_p && \
      CC=/path/to/gcc ../configure --enable-shared && make && sudo make install && cd ..
    mkdir s_p && cd s_p && \
      CC=/path/to/gcc ../configure --enable-shared --enable-single && make && sudo make install && cd ..
    mkdir d_p_omp && cd d_p_omp && \
      CC=/path/to/gcc ../configure --enable-shared --enable-openmp && make && sudo make install && cd ..
    mkdir s_p_omp && cd s_p_omp && \
      CC=/path/to/gcc ../configure --enable-shared --enable-single --enable-openmp && make && sudo make install && cd ..

If you have no `sudo` rights to install FFTW libraries, then provide an installation directory via `make install --prefix="/path/to/fftw"`.

.. note::
    Before building galario, `FFTW_HOME` has to be set equal to the installation directory of FFTW, e.g. with:

    .. code-block:: bash

        export FFTW_HOME="/usr/local/lib/"

    in the default case, or to the prefix specified during the FFTW installation.
    Also, you need to update the `LD_LIBRARY_PATH` to pick the FFTW libraries:

    .. code-block:: bash

        export LD_LIBRARY_PATH=$FFTW_HOME/lib:$LD_LIBRARY_PATH


To speedup building FFTW, you may add the -jN flag to the make commands above, e.g. `make -jN`, where N is an integer
equal to the number of cores you want to use. E.g., on a 4-cores machine, you can do `make -j4`. To use -j4 as default, you can
create an alias with:

.. code-block:: bash

    alias make="make -j4"

Setting paths
^^^^^^^^^^^^^

To find FFTW3 in a nonstandard directory, say `$FFTW_HOME`, tell `cmake` about it:

.. code-block:: bash

    cmake -DCMAKE_PREFIX_PATH=${FFTW_HOME} ..

For multiple directories, use a `;` between directories:

.. code-block:: bash

    cmake -DCMAKE_PREFIX_PATH=${FFTW_HOME};/opt/something/else ..

In case the directory with the header files is not inferred correctly:

.. code-block:: bash

    cmake -DCMAKE_CXX_FLAGS="-I${FFTW_HOME}/include" ..

In case the openmp libraries are not in `${FFTW_HOME}/lib`

.. code-block:: bash

    cmake -DCMAKE_LIBRARY_PATH="${FFTW_OPENMP_LIBDIR}" ..

.. _build_details_cuda:

CUDA
~~~~

`cmake` tests for compilation on the GPU with cuda by default **except on Mac
OS**, where version conflicts between the NVIDIA compiler and the C++ compiler
often lead to problems; see for example `this issue
<https://github.com/mtazzari/galario/issues/30>`_.

To manually enable or disable checking for cuda, do

.. code-block:: bash

   cmake -DGALARIO_CHECK_CUDA=0 .. # don't check
   cmake -DGALARIO_CHECK_CUDA=1 .. # check

Timing
~~~~~~
For testing purposes, you can activate the timing features embedded in the code that produce detailed printouts to `stdout` of various
portions of the functions. The times are measured in milliseconds. This feature is OFF by default and can be activated during the configuration stage with

.. code-block:: bash

    cmake -DGALARIO_TIMING=1 ..

Documentation
~~~~~~~~~~~~~

This documentation should be available online `here
<https://mtazzari.github.io/galario/>`_. If you want to build the documentation
locally, from within the `build/` directory run:

.. code-block:: bash

    make docs

which creates output in `build/docs/html`. The `docs` are not built by default, only upon request.

First install the build requirements with

.. code-block:: bash

   conda install sphinx
   pip install sphinx_py3doc_enhanced_theme

within the conda environment in use. This ensures that the
`sphinx` version matches the Python version used to compile
|galario|.
If you still have problems, remove the `CMakeCache.txt`, rerun
`cmake`, and observe which location of `sphinx` is reported in
`CMakeCache.txt`, for example:

.. code-block:: bash

    -- Found Sphinx: /home/myuser/.local/miniconda3/envs/galario3/bin/sphinx-build

The |galario| library needs to be imported when building the documentation (the
import would fail otherwise) to extract docstrings.

To delete the sphinx cache in case the docs don't update as expected

    rm -rf docs/_doctrees/

.. _install_details:

Install
-------

To specify a path where to install the C libraries of |galario| (e.g., if you do not have `sudo` rights to install it in `usr/local/lib`),
do the conventional:

.. code-block:: bash

    cmake -DCMAKE_INSTALL_PREFIX=/path/to/galario/lib ..

and, after building, run:

.. code-block:: bash

    make install

This will install the C libraries of |galario| in `/path/to/galario/`.

.. note::
    By default the C libraries and the Python bindings are installed under the same prefix.
    If you want to install the Python bindings elsewhere, there is an extra cache variable `GALARIO_PYTHON_PKG_DIR` that you can edit with
    `ccmake .` after running `cmake`.


If you are working inside an active conda environment, both the libraries and the python wrapper are installed inside the environment defined by `$CONDA_PREFIX`, e.g.:

.. code-block:: bash

    conda activate myenv
    cmake ..
    make && make install

Example output during the `install` step

.. code-block:: bash

    -- Installing: /path/to/conda/envs/myenv/lib/libgalario.so
    -- Installing: /path/to/conda/envs/myenv/include/galario.h
    ...
    -- Installing: /path/to/conda/envs/myenv/lib/python2.7/site-packages/galario/single/__init__.py

From the environment `myenv` it is now possible to import |galario|.

Uninstall
~~~~~~~~~

After installation, remove all installed files with

.. code-block:: bash

   make uninstall

Tests
-----

After building, just run `ctest -V --output-on-failure` from within the `build/` directory.

Every time `python/test_galario.py` is modified, it has to be copied over to the build directory: only when run there,
`import pygalario` works. The copy is performed in the configure step, `cmake` detects changes so always run `make` first.

`py.test` fails if it cannot collect any tests. This can be caused by C errors.
To debug the testing, first find out the exact command of the test:

.. code-block:: bash

    make && ctest -V

`py.test` captures the output from the test, in particular from C to stderr.
Force it to show all output:

.. code-block:: bash

    make && python/py.test.sh -sv python_package/tests/test_galario.py

By default, tests do not run on the GPU. Activate them by calling `py.test.sh --gpu=1 ...`.
To select a given parametrized test named `test_sample`, just run `py.test.sh -k sample`.

A cuda error such as

.. code-block:: bash

    [ERROR] Cuda call /home/user/workspace/galario/build/src/cuda_lib.cu: 815
    invalid argument

can mean that code cannot be executed on the GPU at all rather than that specific call being invalid.
Check if `nvidia-smi` fails

.. code-block:: bash

    $ nvidia-smi
    Failed to initialize NVML: Driver/library version mismatch


.. LINKS opening in new tabs/windows

.. |NVIDIA_docs| raw:: html

   <a href="http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements" target="_blank">NVIDIA Docs</a>
