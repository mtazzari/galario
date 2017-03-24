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

Where are GreatCMakeCookOff files

    cmake -DGreatCMakeCookOff_DIR=/home/beaujean/software/GreatCMakeCookOff/cmake ..


python environment
------------------

galario should work with both python 2 and 3

    conda create --name galario3 numpy cython

If you want to run the unit test, you need some more packages

    conda install astropy pytest

testing
-------

After building, just run `ctest --output-on-failure` in `build/`.

To compare with pyvfit need python2.7 and directory of pyvfit's static directory

    cmake \
        -DGreatCMakeCookOff_DIR=/home/beaujean/software/GreatCMakeCookOff/cmake \
        -DPython_ADDITIONAL_VERSIONS=2.7 \
        -D/home/beaujean/workspace/protoplanetary/pyvfit/pyvfit/static \
        .. \
        && make && make test
