====================
A quickstart example
====================

Fit a single-wavelength data set
--------------------------------

In this page we show how to use |galario| in some typical use cases, e.g. the fit of some interferometric data sets.

|galario| has been designed to simplify and accelerate the task of computing the synthetic visibilities given a model image, leaving to the user the freedom to choose the most appropriate statistical tool for the parameter space exploration.

For the purpose of these examples we will adopt a Bayesian approach, using Monte Carlo Markov chains to explore the parameter space and to produce a sampling of the posterior probability function. In particular, we will use the MCMC implementation in the `emcee <http://dfm.io/emcee/current/>`_ Python package.

In this page we will show how to fit the mock observations of a protoplanetary disk. In particular, in the example we will analyse mock visibilities of the disk continuum emission at :math:`\lambda=` 1 mm whose synthetized map is shown in this figure:


.. image:: images/disk_example.jpg
    :scale: 90 %
    :alt: Protoplanetary disk continuum map
    :align: center


You can download `here <https://www.ast.cam.ac.uk/~mtazzari/galario/uvtable.txt>`_ an ASCII version of the uv-table used in this example.

1) Import the useful modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   ##### Import modules
   import numpy as np
   import matplotlib.pyplot as plt
   import math
   ##### galario
   from galario.double import get_image_size, chi2Profile # computes the image size required from the (u,v) data , computes a chi2
   from galario import deg, arcsec # for conversions
   ##### Emcee
   from emcee import EnsembleSampler
   import corner
   from multiprocessing import Pool

   ##### Because we don't want each thread to use multiple core for numpy computation.
   #That forces the use of a proper multithreading
   import os
   os.environ["OMP_NUM_THREADS"] = "1"

2) Import the uv table
^^^^^^^^^^^^^^^^^^^^^^

First, let’s import the table containing the interferometric observations. Typically, an interferometric data set can be exported to a table containing the :math:`(u_j, v_j)` coordinates (:math:`j=1,...,M`), the Real and Imaginary part of the complex visibilities :math:`V_{obs,j}`, and their theoretical weight :math:`w_{j}`, for example:

.. code-block:: python

   u  [m]          v  [m]          Re  [Jy]        Im  [Jy]        w
   -------------------------------------------------------------------------
   -155.90093      234.34887        0.01810         0.13799        200.05723
   9.290660        362.97853       -0.05827         0.02820        216.95405
   95.23531        109.22704        0.06314        -0.16727        167.18789
   94.01319        251.97293        0.01974         0.04358        179.69114
   -60.45751       211.33346        0.14856        -0.07756        188.09953
   91.59843        68.94947         0.12741        -0.12871        156.32432
   23.29531        251.71338        0.01568        -0.12316        189.58017
   -135.83509      -29.77982       -0.02017        -0.00010        207.29354
   59.38624        144.99431        0.04759        -0.08606        201.32301
   192.43093       171.57617       -0.02176        -0.02661        208.52030
   -243.91416      76.18790        -0.02306        -0.01430        207.16036
   58.72442        276.64959        0.03325         0.04560        173.15922
   35.56591        111.28235        0.03777        -0.11856        194.83899
   ...             ...             ...             ...             ...

You can download `here <https://www.ast.cam.ac.uk/~mtazzari/galario/uvtable.txt>`_ an ASCII version of the uv-table used in this example.

code:

.. code-block:: python

   ##### load data
   u, v, Re, Im, w = np.require(np.loadtxt("uvtable.txt", unpack=True), requirements='C')

   ##### for conversion
   wavelength = 1e-3  # [m]
   u /= wavelength
   v /= wavelength

The :math:`u_j` and :math:`v_j` coordinates have been converted in units of the observing wavelength, 1 mm in this example. The ``np.require`` command is necessary to ensure that the arrays are C-contiguous as required by |galario| (see `FAQ 1.1 <https://mtazzari.github.io/galario/FAQ.html#faq1-1>`_\ ).

3) Determine and fix the geometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once imported the uv table, we can start using |galario| to compute the optimal image size.

.. code-block:: python

   ##### get size of the image
   nxy, dxy = get_image_size(u, v, verbose=False) # Number of pixel, width of a pixel in rad

the returned values are the number of pixels (\ ``nxy``\ ) and the pixel size (\ ``dxy``\ ) in radians. ``nxy`` and ``dxy`` are chosen to fulfil criteria that ensure a correct computation of the synthetic visibilities. For more details, refer to Sect. 3.2 in `Tazzari, Beaujean and Testi (2017) <https://arxiv.org/abs/1709.06999>`_.

Then we define the mesh we will use to compute the model. Here is a 1D mesh manualy defined and fixed all through the example.

.. code-block:: python

   ##### radial grid parameters, fixed
   Rmin = 1e-4  # arcsec
   dR = 0.005   # arcsec
   nR = 2000

   ##### convert it to use it with galario.double.chi2Profile
   dR *= arcsec
   Rmin*=arcsec

   ##### Define a mesh for the space
   R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)

Defining the mesh out of the functions, as a global constant, makes the computation faster for larger examples. Yet you might need to pass it as an argument, in which case you should refer to the documentation of `emcee <http://dfm.io/emcee/current/>`_.

4) Define the brightness model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us define the model: for this example, we will use a very simple Gaussian profile:

.. code-block:: python

   ##### Define a gaussian profile
   def GaussianProfile(f0, sigma):
       """ Gaussian brightness profile.
       """
       return( f0 * np.exp(-(0.5/(sigma**2.))*(R**2.) ))

``f0`` (Jy/sr) is a normalization, ``sigma`` is the width of the Gaussian, and ``R`` is the globaly defined mesh.

5) Setup the MCMC Ensemble Sampler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In our fit we will have 6 free parameters: on top of the model parameters ``f0`` and ``sigma`` we want to fit the inclination ``inc``, the position angle ``PA``, and the angular offsets (\ ``dRA`` and\ ``dDec``\ ) with respect to the phase center. Following the notation of the `emcee <http://dfm.io/emcee/current/>`_ documentation, we initialise the EnsembleSampler.

``p_range`` is a rectangular domain in the parameter space that defines the search region.

.. code-block:: python

   ##### Initialise the "first guess"
   p0 = np.array([10., 0.5, 70., 60., 0.1, 0.1]) #  2 parameters for the model + 4 (inc, PA, dRA, dDec)

   ##### parameter space domain: the parameters can't go out of these
   p_range = np.array([[1., 20.],  #f0
               [0., 8.],           #sigma
               [0., 90.],          #inc
               [0., 180.],         #pa
               [-2., 2.],          #dra
               [-2., 2.]])         #ddec

   ##### define emcee parameters
   ndim       = len(p_range)       # number of parameters to fit
   nwalkers   = 40                 # number of walkers (at least twice ndim)
   nthreads   = 4                  # CPU threads that emcee should use
   iterations = 3000               # total number of MCMC steps

   ##### initialize the walkers with an ndim-dimensional ball
   pos = np.array([(1. + 1e-4 * np.random.random(ndim)) * p0 for i in range(nwalkers)])

6) Define the posterior and the prior probability functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We now need to define a likelyhood for our model, a way to evaluate how close to the data it is. For that, we implement the posterior function, using galario to compute the :math:`\chi^2`.

Since in this example we are assuming an axisymmetric brightness profile we will use the ``chi2Profile`` function, but the same design holds for the ``chi2Image`` function that should be used for non-axisymmetric profiles.

First we need to ensure we stay in the boundaries we fixed.

.. code-block:: python

   def lnpriorfn(p):
       # if we are out of range
       if np.any(p<p_range[:,0]) or np.any(p>p_range[:,1]):
           return(-np.inf)
       return(0.)

And then we implement the full cost function, using a conversion for the units of ``chi2Profile``\ , and a logarithmic value for ``f0`` as it speeds up the convergence.

.. code-block:: python

   ##### Define a conversion to translate the data for galario.double.chi2Profile
   def convertp(p):
           f0, sigma, inc, PA, dRA, dDec = p
           return(10.**f0, sigma*arcsec, inc*deg, PA*deg, dRA*arcsec, dDec*arcsec)

   ##### Define the cost
   def lnpostfn(p):
       """ Log of posterior probability function """
       # test if we are in the boundaries
       lnp = lnpriorfn(p)
       if not np.isfinite(lnp):
           return -np.inf
       # unpack the parameters
       f0, sigma, inc, PA, dRA, dDec = convertp(p)
       # compute the model brightness profile
       f = GaussianProfile(f0, sigma)
       # compute the cost
       chi2 = chi2Profile(f, Rmin, dR, nxy, dxy, u, v, Re, Im, w, inc=inc, PA=PA, dRA=dRA, dDec=dDec)
       return(-0.5 * chi2)

7) Launching the MCMC process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

According to your version of emcee:


* Version 3 (with progress bar)

.. code-block:: python

   ##### execute the MCMC
   with Pool(processes=nthreads) as pool:
       sampler = EnsembleSampler(nwalkers, ndim, lnpostfn,pool=pool)
       pos, prob, state = sampler.run_mcmc(pos, iterations, progress=True)

* Version 2 (conda's default)

.. code-block:: python

   sampler = EnsembleSampler(nwalkers, ndim, lnpostfn,threads=nthreads)
   pos, prob, state = sampler.run_mcmc(pos, iterations)

8) Plot the optimization
^^^^^^^^^^^^^^^^^^^^^^^^

You can see the advancement of each parameter and check their convergence using these few lines.

.. code-block:: python

   samples = sampler.chain

   ##### Get the shape of the plot
   nwalkers,iterations,ndims = samples.shape
   ncols = 2
   nrows = 3

   ##### labeling
   labels=[r"$f_0$", r"$\sigma$", r"$Inc$", r"PA", r"$\Delta$RA", r"$\Delta$Dec"]

   ##### Make a figure
   fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=(15, 10), sharex=True)
   for i in range(ndims):
       ax = axes.flatten()[i]
       _=ax.plot(np.transpose(samples[:, :, i]), "k", alpha=0.3)
       _=ax.set_xlim(0, iterations)
       _=ax.set_ylabel(labels[i])
   #    _=ax.yaxis.set_label_coords(-0.1, 0.5)
   #    _=ax.plot([0,iterations],[p_range[i,0],p_range[i,0]])
   #    _=ax.plot([0,iterations],[p_range[i,1],p_range[i,1]])

   _=ax.set_xlabel('iterations')
   plt.tight_layout()
   plt.show()


.. image:: images/advancingplot.jpg
    :scale: 90 %
    :alt: Evolution of the emcee parameters
    :align: center


It is possible to run the whole fit collecting the code blocks above into a single ``quickstart.py`` file and running ``python quickstart.py``. For reference, using ``nthreads=4``\ , the run takes approximately 5 minutes on a laptop with an Intel Core i5 @ 2.9GHz.

9) Plot the fit results
^^^^^^^^^^^^^^^^^^^^^^^

You now can plot the corelation between each parameter, using a corner plot.

.. code-block:: python

   ##### Reshape on the converged zone
   cornering=(samples[:,-1000:,:].reshape((-1,ndims)))

   ##### plot
   fig = corner.corner(cornering,
       quantiles=[0.16, 0.50, 0.84],
       labels=labels,
       show_titles=True,
       label_kwargs={'labelpad':20, 'fontsize':0},
       fontsize=8)

   fig.show()
