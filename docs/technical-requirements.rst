======================
Technical requirements
======================

In this page we summarize the assumptions that are made in |galario| and the requirements on the input quantities.

Full details are presented in Section 3 of Tazzari, Beaujean and Testi (2018) MNRAS **476** 4527
`[MNRAS] <https://doi.org/10.1093/mnras/sty409>`_
`[arXiv] <https://arxiv.org/abs/1709.06999>`_
`[ADS] <http://adsabs.harvard.edu/abs/2018MNRAS.476.4527T>`_.

..    Assumptions
    - **small-field imaging** the first release of |galario| computes visibilities,
    thus neglecting the non-coplanarity of the baselines.
    This restricts the usage of the code to the cases in which the the region modelled
    with \func{*Image} or \func{*Profile} lies within the region defined in Eq.~\eqref{eq:wprojection.limit}.
    - **Primary-beam correction** the `*Image` functions take as input an image of
    the primary-beam corrected brightness :math:`\mathcal{A}I_\nu(l,m)`.
    In the cases in which the region of interest in the image plane is small compared to
    the primary beam and close to its centre, one can approximate
    :math:`\mathcal{A}I_\nu\approx I_\nu` and apply the \func{*Image} functions directly
    to the brightness without significant deviations.
    The choice whether to apply this approximation is left to the user.
    We note, however, that in the first released version of the code
    the \func{*Profile} functions --- which take as input a profile $I_\nu(R)$ and
    internally compute $I_\nu(l,m)$ --- do not apply the primary beam correction.
    - **Frequency dependence** of $\mathcal{A}$ and $I_\nu$: both the antenna pattern
    and the source brightness are frequency-dependent quantities.
    As stated in the previous Section, the definition in
    Eq.~\eqref{eq:complex.visibility.obs} holds for small bandwidths $\Delta \nu$
    over which the integrand can be assumed constant. For this reason, in the first release
    of \galario, the visibilities are assumed all at the same average frequency $\nu_0$.
    This implies that, in order to compare synthetic visibilities to observed ones
    (e.g. through Eq.~\eqref{chap6.eq:def.chi.square} with the \func{chi2*} functions),
    the observed visibilities (typically consisting of multiple measurements over several
    hundreds of spectral channels) must be channel-averaged\footnote{{This can be achieved,
    e.g., with the \comm{split} command of the Common Astronomy Software Application (CASA) package.}}
    into a single channel at frequency $\nu_0$ and characterised by a small $\Delta \nu$.
    We note that the effect of channel averaging is to combine the brightness measurements over a region
    with angular extent $\frac{\Delta\nu}{\nu_0}\sqrt{l^2+m^2}$ along the radial direction.
    Often termed \textit{bandwidth smearing}, this effect is not negligible at the
    distances $\sqrt{l^2+m^2}$ where its angular extent becomes comparable with the synthesized beam.
    The user can choose $\Delta\nu$ in order to control the bandwidth smearing within
    the image plane region of interest.
    The computation of synthetic visibilities of a field of view with multiple
    sources can be done in basically two ways: either by applying \func{*Image}
    to an image of $\mathcal{A}I_\nu(l,m)$ containing all the sources, or by
    summing up the visibilities of each single source computed independently
    with either \func{*Image} or \func{*Profile}.
    In the second approach, the displacement of each source in the field of view
    can be achieved (at a small computational cost) by applying a different
    complex phase to the individual visibilities as described in the next Section.
    While the first approach requires executing only one Fourier transform
    --- appearing theoretically more computationally convenient ---
    the second approach exploits the linearity of the Fourier transform and
    might yield results faster if there are many identical sources to be placed
    in different locations.
    It is worth highlighting that in all cases (single or multiple sources
    in the field of view), the limitations due to the assumptions (i) to (iii) apply:
    all the sources must be located in a region that is close to the phase centre
    and small compared to $\theta_{\mathrm{F}}$ and the synthetic visibilities are
    computed in a narrow band around the observing frequency $\nu_0$.

.. _technical_requirements_image_specs:

Image specifications
--------------------
Following the Figure below, there are two fundamental coordinate systems that define the input image for the
:func:`sampleImage() <galario.double.sampleImage>` and :func:`chi2Image() <galario.double.chi2Image>` functions:

    - the **matrix axes** :math:`[i, j]` mapping the pixel coordinates, running from `0` to `Nxy-1` (`Nxy` is the number
      of pixels on each axis).

    - the **physical axes** :math:`(R.A., Dec.)` mapping Right Ascension and Declination coordinates.

The origin `[i, j] = [0, 0]` of the matrix axes can be put either in the **upper left** or in the **lower left** corner of the matrix.
By default |galario| assumes the origin is in the **upper left** corner of the matrix, but it can be changed to the
lower corner by specifying the optional parameter `origin='lower'` in the :func:`sampleImage() <galario.double.sampleImage>`
and :func:`chi2Image() <galario.double.chi2Image>` functions.

The origin :math:`(R.A., Dec.)=(0,0)` of the physical axes is always in the center of the `[Nxy/2, Nxy/2]` pixel
(grey pixel in the Figure below) for any value of `origin`. The :math:`R.A.` axis always increases leftwards, following the usual convention
of having East to the left and West to the right: therefore, the :math:`R.A.` axis always decreases with increasing `j` index.
Note that, with `origin='upper'`, both the :math:`(R.A., Dec.)` axes **decrease** with increasing `i`, `j` index (see Figure below).

.. warning::

    |galario| assumes that the values of the input image are evaluated in the **pixel centers** (not in the pixel edges).

    For instructions on how to compute the correct :math:`(R.A., Dec.)` coordinate meshgrid to create the image,
    see the :ref:`Cookbook <cookbook_meshgrid>`.

The left and right panels of the Figure below show the relative orientations of matrix and physical axes
for the `origin='upper'` and `origin='lower'` cases, respectively (click on the images for a larger version).

    +------------------------------------------------------+-------------------------------------------------------+
    |.. image:: images/galario_image_origin_upper.png      | .. image:: images/galario_image_origin_lower.png      |
    |  :width:  700 px                                     |     :width: 700 px                                    |
    |  :alt: sketch origin upper                           |     :alt: sketch origin lower                         |
    +------------------------------------------------------+-------------------------------------------------------+

.. note::

    The `origin` parameter in |galario| follows the same definition as in the `matshow` and `imshow` commands of the
    `matplotlib` library.
    To get the desired orientation of the Declination axis in |galario|, use the same `origin` parameter that produces
    the desired image orientation with `matshow` or `imshow`.