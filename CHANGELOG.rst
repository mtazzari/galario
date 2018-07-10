1.2 (2018-06-20)
++++++++++++++++

- [interface] New `origin` option to specify direction of Dec axis for input image in `*Image()` functions.
- [core/bugfix] More robust DFT interpolation for sources that are large or hugely offset from phase center.
- [docs] New `Tech specs` page on specifications of the input image with new `origin` option.
- [docs] New `Cookbook` recipe on how to create the correct coordinate mesh grid.

1.1 (2018-05-14)
++++++++++++++++

- [interface] Python and C++ code now throw detailed exceptions allowing fine-grained control, e.g. for executions on GPU.
- [core/bugfix] More robust interpolation of brightness profile in the central pixel for steep `f(R)` profiles.
- [core] Drop support for C. Only C++ and Python are now supported.
- [core] Memory handling on GPU: memory is now automatically freed in case of an error (allows catching errors with Exceptions).

1.0.2 (2017-12-19)
++++++++++++++++++
- [interface] CPU version can now be installed with `conda install -c conda-forge galario`.
- [core/bugfix] Fix memory leak in GPU version.
- [core] Allow multiple processes to use the GPU concurrently by default.
- [docs] Improve installation notes and quickstart example.

1.0.1 (2017-10-05)
++++++++++++++++++
- [interface] Allow uninstalling galario with `make uninstall`.
- [interface] Allow enabling/disabling check for CUDA on Mac OS with `cmake -DGALARIO_CHECK_CUDA=1`.

1.0 (2017-09-19)
++++++++++++++++
- First release.
