.. :FAQ:

.. default-role:: code
.. role:: bash(code)
   :language: bash

.. _FAQ:

=================================================
|galario| Frequently Asked Questions with Answers
=================================================

This is the list of Frequently Asked Questions about |galario|. The list will be updated regularly to include the questions about recurring issues and relative solutions.

In all the code snippets shown in these FAQs, wherever `np` is present is defined as  `import numpy as np`.


Section 1 - Using |galario|
--------------------------------------------

.. _FAQ1.1:

**Question 1.1. Why do I get an error about non C-contiguous arrays?**

I run this line (or a similar one with `chiImage()`, `sampleProfile()` or `sampleImage()`):

.. code-block:: python

    chi2 = chi2Profile(f, Rmin, dR, nxy, dxy, u, v, Re, Im, w)


and I get the following error:

.. code-block:: bash

  File "libcommon.pyx", line 630, in libcommon.chi2Profile
  File "stringsource", line 653, in View.MemoryView.memoryview_cwrapper
  File "stringsource", line 348, in View.MemoryView.memoryview.__cinit__
  ValueError: ndarray is not C-contiguous

The issue is caused by one of the arrays passed in input to `chi2Profile` being not C-contiguous, which is a requirement for the C++ code in GALARIO. You can check whether a NumPy array `x` is C-contiguous by printing `x.flags`. The first action to debug this issue is to print the flags of all the arrays in input to the function.

Typically this happens with the `u`, `v`, `Re`, `Im`, `w` arrays that are not C-contiguous if you read them, e.g., from an ASCII uvtable with a `np.loadtxt()` command and the `unpack=True` option (or something equivalent).

**SOLUTION:** Make the arrays C-contiguous with the `np.ascontiguousarray()` command. Applying this to the example:

.. code-block:: python

    u = np.ascontiguousarray(u)
    v = np.ascontiguousarray(v)
    Re = np.ascontiguousarray(Re)
    Im = np.ascontiguousarray(Im)
    w = np.ascontiguousarray(w)

Alternatively, you can make the arrays contiguous all at once:

.. code-block:: python

    u, v, Re, Im, w = np.require([u, v, Re, Im, w], requirements='C')



..
    I get this error:
    ImportError                               Traceback (most recent call last)
    <ipython-input-144-ee2f01adc0c4> in <module>()
    ----> 1 from galario.double_cuda import sampleImage
    /Users/tdavis/anaconda/lib/python3.5/site-packages/galario/double_cuda/__init__.py in <module>()
    ----> 1 from .libcommon import *
          2
          3 _init()
          4
          5 import atexit
    ImportError: No module named 'galario.double_cuda.libcommon'

..
    2) I am mainly interested in using this for line cubes.
    Is the best way to do that to loop over and do each channel separately?
    Does that add overhead? Any way to cut that down?

..
    I have a single source, far from the center, what's the best way of modelling it?
    sampleProfile with dRa, dDec or sampleImage?
    Anything to pay attention to? -> nxy, dxy

    How do I check if nxy, dxy chosen are correct?

    How can I use more than one GPU?
