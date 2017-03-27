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

Specify a python version. In `build/`, do

    cmake -DPython_ADDITIONAL_VERSIONS=3.5 ..

For now we depend on GreatCMakeCookOff, clone it from https://github.com/UCL/GreatCMakeCookOff and let cmake know where it is, for example

    cmake -DGreatCMakeCookOff_DIR=/home/beaujean/software/GreatCMakeCookOff/cmake ..

If we decide to keep this dependency, we can include it into the cmake configure step and it is downloaded automatically.

python environment
------------------

galario should work with both python 2 and 3

    conda create --name galario2 python=2 numpy cython
    conda create --name galario3 python=3 numpy cython

If you want to run the unit test, you need some more packages

    conda install astropy pytest

cmake may get confused with the conda python and the system
python. This is a general problem
https://cmake.org/Bug/view.php?id=14809

A workaround to help cmake find the interpreter and the libs from the
conda environment is

    cmake -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} ..

testing
-------

After building, just run `ctest --output-on-failure` in `build/`.

To compare with pyvfit need `python2.7` and `pyvfit` installed in
developer mode so we can pick up the `static` directory from the
location of `pyvfit` itself.

My current one liner to get going is

    cmake \
        -DGreatCMakeCookOff_DIR=$HOME/workspace/GreatCMakeCookOff/cmake \
        -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
        .. \
    && make && make test
