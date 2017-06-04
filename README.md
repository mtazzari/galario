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

Bug to be fixed: before doing make, you have to manually create the directory:
    
    mkdir galario/docs/_static

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

    FFTW_HOME="/usr/local/lib" CXX="/usr/local/bin/g++" CC="/usr/local/bin/gcc" LD="/usr/local/bin/g++" \
    cmake -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} .. && make all test

Every time `python/test_all.py` is modified, it has to be copied over
to the build directory: only when run there, `import pygalario`
works. The copy is performed in the build step but I couldn't get the
dependency injected, so to run the tests, you have to do `make && make
test` or `make && ctest`.

requirements
------------

The FFTW libraries are required for the CPU version of galario. 
To install FFTW follow the instructions at http://www.fftw.org. 
galario requires the following FFTW libraries:
    
    * libfftw3              # double precision
    * libfftw3f             # single precision
    * libfftw3_omp          # double precision with OpenMP
    * libfftw3f_omp         # single precision with OpenMP
    * libfftw3_threads      # double precision with threads
    * libfftw3f_threads     # single precision with threads

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
    mkdir d_p_threads && cd d_p_threads && \
      CC=/usr/local/bin/gcc ../configure --enable-shared --enable-threads && make && sudo make install && cd ..
    mkdir s_p_threads && cd s_p_threads && \
      CC=/usr/local/bin/gcc ../configure --enable-shared --enable-single --enable-threads && make && sudo make install && cd ..

If you have no sudo rights to install FFTW libraries, then provide a directory via make install --prefix="/path/to/fftw".
Before building galario, FFTW_HOME has to be set equal to the installation directory of FFTW, e.g. FFTW_HOME="/usr/local/lib/"
in the default case, or to the prefix specified during the fftw installation.

To speedup building FFTW, you may add the -jN flag to the make commands above, e.g. make -jN, where N is an integer 
equal to the number of cores you want to use. E.g., on a 4-cores machine, you can do make -j4. To use -j4 as default, you can 
create an alias with:
    
    alias make="make -j4"

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

documentation
-------------

    make docs

creates output in `docs/html` under the build directory. Add content to
`docs/index.rst` or the files linked to therein.
