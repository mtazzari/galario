galario
=======

**Gpu Accelerated Library for Analysing Radio Interferometer Observations**

[![Release Number](https://img.shields.io/github/release/mtazzari/galario/all.svg)](https://github.com/mtazzari/galario/releases)
[![Build Status](https://travis-ci.org/mtazzari/galario.svg?branch=master)](https://travis-ci.org/mtazzari/galario)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![DOI](https://zenodo.org/badge/82575704.svg)](https://zenodo.org/badge/latestdoi/82575704)
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/galario.svg)](https://anaconda.org/conda-forge/galario)

**galario** is a library that exploits the computing power of modern graphic cards (GPUs) to accelerate the comparison of model
predictions to radio interferometer observations. Namely, it speeds up the computation of the synthetic visibilities
given a model image (or an axisymmetric brightness profile) and their comparison to the observations.

Check out the [documentation](https://mtazzari.github.io/galario/) and the [installation instructions](https://mtazzari.github.io/galario/install.html).

If you use **galario** for your research please cite  Tazzari, Beaujean and Testi (2018) MNRAS **476** 4527 [[MNRAS]](https://doi.org/10.1093/mnras/sty409) [[arXiv]](https://arxiv.org/abs/1709.06999) [[ADS]](http://adsabs.harvard.edu/abs/2018MNRAS.476.4527T):
```
@ARTICLE{2018MNRAS.476.4527T,
   author = {{Tazzari}, M. and {Beaujean}, F. and {Testi}, L.},
    title = "{GALARIO: a GPU accelerated library for analysing radio interferometer observations}",
  journal = {\mnras},
archivePrefix = "arXiv",
   eprint = {1709.06999},
 primaryClass = "astro-ph.IM",
 keywords = {methods: numerical, techniques: interferometric, submillimetre: general},
     year = 2018,
    month = jun,
   volume = 476,
    pages = {4527-4542},
      doi = {10.1093/mnras/sty409},
   adsurl = {http://adsabs.harvard.edu/abs/2018MNRAS.476.4527T},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


License
-------
**galario** is free software licensed under the LGPLv3 License. For more details see the LICENSE.

© Copyright 2017-2018 Marco Tazzari, Frederik Beaujean, Leonardo Testi.
