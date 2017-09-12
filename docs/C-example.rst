.. http://www.sphinx-doc.org/en/stable/domains.html#the-c-domain

.. default-domain:: c

====================
Using galario from C
====================

The core of galario is accessible from C, all functions are in the header `galario.h`. Of course it can be used from most other languages as well, in particular C++.

A small example
---------------

Here is a small test program that performs the FFT in 2D on an image with random values

.. literalinclude:: ../src/galario_test.c
                    :language: c

After successfully installing galario to `/path/to/galario`, a simple
test program running galario on the CPU with `openMP` and double
precision can be built with::

  gcc -I/path/to/galario/include -L/path/to/galario/lib -lgalario -DDOUBLE_PRECISION galario_test.c -o galario_test

To use single precision, simply do not define the preprocessor symbol `-DDOUBLE_PRECISION` and link in the appropriate library with `-lgalario_single`. A mismatch between the library and the preprocessor symbol causes undefined behavior but usually a segmentation fault causes the program to abort at runtime.

If galario was installed with `cuda` support, link in `-lgalario_cuda` or `-lgalario_single_cuda`.

Example walk-through
--------------------

A small walk through the example's `main` function line by line::

  galario_init();
  ...
  galario_cleanup();

Before any computation is done inside galario, the library has to be initialized. Similarly, any data created during initialization should be cleaned at the end of `main`.

To create the input image, define an array::

   dreal realdata[nx*ny];

The data type `dreal` can refer to either `float` or `double`, depending on the preprocessor symbol `DOUBLE_PRECISION`. galario assumes the input is a real image but the output of the FFT is complex. galario provides a helper function to allocate an array of the proper size and to copy over the input image::

  dcomplex* res = galario_copy_input(nx, ny, realdata);

The actual FFT is done in-place, and the result is stored in `res`. The data layout is described in the `FFTW manual <http://fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data>`_::

  galario_fft2d(nx, ny, res);

To deallocate `res`, we use::

  galario_free(res);

In general, any array created by galario and handed back to the user must be deallocated using `galario_free`.