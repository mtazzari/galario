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

Specify a python version. In `build/`, do

    rm CMakeCache.txt; cmake -DPython_ADDITIONAL_VERSIONS=3.5 ..


python environment
------------------

galario should work with both python 2 and 3

    conda create --name galario3 numpy cython

If you want to run the unit test, you need some more packages

    conda install astropy pytest

testing
-------

After building, just run `ctest --output-on-failure` in `build/`.
