
====================
A quickstart example
====================

The CPU version of `galario` is always compiled, also on a system without a CUDA-enabled GPU. In this case you can start using the double- or single-precision CPU versions with:

.. code-block:: python

    from galario.double import ...
    from galario.single import ...

If built on a machine with a CUDA-enabled GPU, `galario` is compiled both for the CPU **and** the GPU. You can start using the GPU version as simply as:

.. code-block:: python

    from galario.double_cuda import ...
    from galario.single_cuda import ...

TBC