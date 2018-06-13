.. http://www.sphinx-doc.org/en/stable/domains.html#the-c-domain

.. default-domain:: c++

======================
Using galario from C++
======================

The core of |galario| is written in C++/Cuda, all functions are in the header `galario.h`.

A small example
---------------

Here is a small test program that performs the FFT in 2D on an image with random values. It is part of the |galario| test suite.

.. literalinclude:: ../src/galario_test.cpp
                    :language: c++

After successfully installing |galario| to `/path/to/galario`, a simple
test program running |galario| on the CPU with `openMP` and double
precision can be built with::

  g++ -I/path/to/galario/include -L/path/to/galario/lib -lgalario -DDOUBLE_PRECISION galario_test.c -o galario_test

To use single precision, simply do not define the preprocessor symbol `-DDOUBLE_PRECISION` and link in the appropriate library with `-lgalario_single`. A mismatch between the library and the preprocessor symbol causes undefined behavior but usually a segmentation fault causes the program to abort at runtime.

If |galario| was installed with `cuda` support, you can link in `-lgalario_cuda` or `-lgalario_single_cuda` instead.

Example walk-through
--------------------

A small walk through the example's `main` function line by line::

   #include "galario.h"
   ...
   using namespace galario;

All |galario| functions are declared in the header `galario.h` and inside the `namespace galario`. If you want to use particular functions without making all names in the `namespace` available, use for example  `galario::sweep`::

  init();
  ...
  cleanup();

Before any computation is done inside |galario|, the library has to be initialized. Similarly, any data created during initialization should be cleaned at the end of `main()`.

To create the input image, define a `std::vector` of the appropriate size. Not that this initializes all values to zero::

   std::vector<dreal> realdata(nx*ny);

The data type `dreal` can refer to either `float` or `double`, depending on the preprocessor symbol `DOUBLE_PRECISION`. |galario| assumes the input is a real image but the output of the FFT is complex. |galario| provides a helper function to allocate an array of the proper size and to copy over the input image::

  dcomplex* res = copy_input(nx, ny, realdata);

The actual FFT is done in-place, and the result is stored in `res`. The data layout is described in the `FFTW manual <http://fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data>`_ and applied to the CPU and GPU versions of |galario|::

  fft2d(nx, ny, res);

To deallocate `res`, we use::

  galario_free(res);

In general, any array created by |galario| and handed back to the user must be deallocated using `galario_free`. The reason is that in the CPU version, we use the FFTW to allocate memory that is aligned to allow FFTW to be most efficient (think SIMD vectorization). Correspondingly, we should not use the standard C library's `free` function.
