import numpy as np
import os
import pytest

import galario

if galario.HAVE_CUDA and int(pytest.config.getoption("--gpu")):
    from galario import double_cuda as g_double
    from galario import single_cuda as g_single
else:
    from galario import double as g_double
    from galario import single as g_single

# PARAMETERS FOR MULTIPLE TEST EXECUTIONS
par1 = {'wle_m': 0.003, 'x0_arcsec': 0.4, 'y0_arcsec': 4.}
par2 = {'wle_m': 0.00088, 'x0_arcsec': -3.5, 'y0_arcsec': 7.2}
par3 = {'wle_m': 0.00088, 'x0_arcsec': 0., 'y0_arcsec': 0.}

sec2rad = np.pi/180./3600.  # from arcsec to radians

# use last gpu if available. Check `watch -n 0.1 nvidia-smi` to see which gpu is
# used during test execution.
ngpus = g_double.ngpus()
g_double.use_gpu(max(0, ngpus-1))

g_double.threads_per_block()


########################################################
#                                                      #
#                  REFERENCE FUNCTIONS                 #
#                                                      #
########################################################
def create_reference_image(size, x0=10., y0=-3., sigma_x=50., sigma_y=30., dtype='float64', reverse_xaxis=False, correct_axes=True, **kwargs):
    """
    Creates a reference image: a gaussian brightness with elliptical
    """
    _ = kwargs.get('kernel', 0.)  # legacy: muted
    _ = kwargs.get('save', 0.)  # legacy: muted

    inc_cos = np.cos(0./180.*np.pi)

    delta_x = 1.
    x = (np.linspace(0., size-1, size) - size/2.) * delta_x


    if reverse_xaxis:
        xx, yy = np.meshgrid(-x, x/inc_cos)
    elif correct_axes:
        xx, yy = np.meshgrid(-x, -x/inc_cos)
    else:
        xx, yy = np.meshgrid(x, x/inc_cos)

    image = np.exp(-(xx-x0)**2./sigma_x - (yy-y0)**2./sigma_y)

    return image.astype(dtype)


def create_reference_image_slow(size=1024, kernel='gaussian', save=False, dtype='float64'):
    try:
        import astropy
    except ImportError:
        print("Please install astropy with:\n\tconda install astropy")
        exit(1)

    from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
    from astropy.modeling.models import Gaussian2D

    np.random.seed(42)

    gauss = Gaussian2D(1, 0, 0, 3, 3)

    # Fake image data including noise
    x = np.linspace(-100, 100, size)
    y = np.linspace(-100, 100, size)
    x, y = np.meshgrid(x, y)
    data_2D = gauss(x, y) + 0.1 * (np.random.rand(size, size) - 0.5)

    if kernel == 'tophat':
        kernel = Tophat2DKernel(30)
    elif kernel == 'gaussian':
        kernel = Gaussian2DKernel(2)

    reference_image = convolve(data_2D, kernel)

    # ensure it is positive
    reference_image += np.abs(reference_image.min())

    if save:
        import matplotlib.pyplot as plt
        plt.imshow(reference_image)
        plt.savefig(os.path.join("./", 'reference_image.pdf'))

    return reference_image.astype(dtype)


def create_sampling_points(nsamples, maxuv=1., dtype='float64'):
    np.random.seed(42)

    # columns are non contiguous arrays => copy
    x = np.random.uniform(low=-maxuv, high=maxuv, size=(nsamples, 2))
    return x[:, 0].astype(dtype), x[:, 1].astype(dtype)


def uv_idx(udat, vdat, uv):
    """
    uv coordinates to pixel coordinates in range [0, npixels].
    Assume image is square, same boundary in u and v direction.

    Parameters
    ----------

    uv: nd array
    u values at which FFT is computed. Assumed identical for v.

    """
    umin = uv[0]
    du = uv[1] - uv[0]

    u = np.floor((udat - umin) / du)
    v = np.floor((vdat - umin) / du)

    return u + (udat - uv) / du , v + (vdat - uv) / du


def pixel_coordinates(maxuv, nx):
    """
    Compute the array that maps the pixels of the image to real uv-coordinates.
    The array contains the coordinate of the pixel centers (not the edges!).

    """
    # TODO: should divide by /nx instead of /(nx-1)

    return (np.linspace(0., nx-1, nx) - nx/2.) * maxuv/(nx)


def get_uv_idx_n(ux, vx, ur, vr, size):

    ntot = len(ur)
    assert len(ur) == len(vr)
    uri = np.zeros(len(ur), dtype=ur.dtype)
    vri = np.zeros(len(vr), dtype=vr.dtype)

    for i in range(ntot):
        i2u = size-1
        i1u = 0
        # binary search: index of closest u element
        while i2u-i1u > 1:
            itu = i1u + int(np.real(i2u-i1u)/2.)
            if ux[itu] > ur[i]:
                i2u = itu
            else:
                i1u = itu

        i2v = size-1
        i1v=0
        while i2v-i1v > 1:
            itv=i1v+int(np.real(i2v-i1v)/2.)
            if vx[itv] > vr[i]:
                i2v = itv
            else:
                i1v = itv

        uri[i] = i1u + np.real(ur[i]-ux[i1u])/(ux[i2u]-ux[i1u])
        vri[i] = i1v + np.real(vr[i]-vx[i1v])/(vx[i2v]-vx[i1v])

    return uri, vri


def int_bilin(f, x, y):

    nd = len(x)

    fint = np.zeros(nd, dtype=f.dtype)

    for i in range(nd):
        jj = int(x[i])
        ii = int(y[i])
        dfj = f[ii + 1, jj] - f[ii, jj]           # x
        dfj1 = f[ii + 1, jj + 1] - f[ii, jj + 1]  # y
        # numpy has weird promotion rules. Use `trunc` instead of `int` to preserve types of `x` and `f`
        fix = f[ii, jj] + dfj * (x[i] - np.trunc(x[i]))
        fix1 = f[ii + 1, jj] + dfj1 * (x[i] - np.trunc(x[i]))
        fint[i] = fix + (fix1 - fix) * (y[i] - np.trunc(y[i]))

    return fint


def matrix_size(udat, vdat, **kwargs):

    maxuv_factor = kwargs.get('maxuv_factor', 4.8)
    minuv_factor = kwargs.get('minuv_factor', 4.)

    uvdist = np.sqrt(udat**2 + vdat**2)

    maxuv = max(uvdist)*maxuv_factor
    minuv = min(uvdist)/minuv_factor

    minpix = np.uint(maxuv/minuv)

    Nuv = kwargs.get('force_nx', int(2**np.ceil(np.log2(minpix))))

    return Nuv, minuv, maxuv


def Fourier_shift_static(ft_centered, x0, y0, wle, maxuv):
    """
    Performs a translation in the real space by applying a phase shift in the Fourier space.
    This function applies the shift to 2D arrays (i.e. images).

    Parameters
    ----------
    ft_centered: 2D float array, complex64
        Fourier transform
    x0, y0: floats, arcsec
        Shifts in the real space.

    Returns
    -------
    v_shifted: 2D float array, complex64
        Phase-shifted Fourier transform

    """
    nx = ft_centered.shape[0]
    # convert x0, y0 from arcsec to pixel

    sec2pixel = sec2rad/wle
    x0 *= sec2pixel
    y0 *= sec2pixel

    # construct the phase change
    spatial_freq = maxuv*np.fft.fftshift(np.fft.fftfreq(nx))*2.*np.pi
    uu, vv = np.meshgrid(spatial_freq, spatial_freq)
    uv_grid = uu*x0 + vv*y0
    cos_theta = np.cos(uv_grid)
    sin_theta = np.sin(uv_grid)

    # apply the phase change
    re_ft_c, im_ft_c = ft_centered.real, ft_centered.imag
    re_v_shifted = re_ft_c*cos_theta - im_ft_c*sin_theta
    imag_v_shifted = im_ft_c*cos_theta + re_ft_c*sin_theta

    v_shifted = re_v_shifted+1j*imag_v_shifted

    return v_shifted


def Fourier_shift_array(u, v, fint, x0, y0):
    """
    Performs a translation in the real space by applying a phase shift in the Fourier space.
    This function applies the shift to data points sampling the Fourier transform of an image.

    Parameters
    ----------
    u, v: 1D float array
        Coordinates of points in the Fourier space. units: observing wavelength
    fint: 1D float array, complex
        Fourier Transform sampled in the (u, v) points.
        Re, Im, u, v must have the same length.
    x0, y0: floats, arcsec
        Shifts in the real space.

    Returns
    -------
    fint_shifted: 1D float array, complex
        Phase-shifted of the Fourier Transform sampled in the (u, v) points.

    """
    # convert x0, y0 from arcsec to cm
    x0 *= sec2rad
    y0 *= sec2rad

    x0 *= 2.*np.pi
    y0 *= 2.*np.pi

    # construct the phase change
    theta = u*x0 + v*y0

    # apply the phase change
    fint_shifted = fint * (np.cos(theta) + 1j*np.sin(theta))

    return fint_shifted


def generate_random_vis(nsamples, dtype):
    x = 3. * np.random.uniform(low=0., high=1., size=nsamples).astype(dtype) + 2.8 +\
        1j * np.random.uniform(low=0., high=1., size=nsamples).astype(dtype) + 8.2
    y = 8. * np.random.uniform(low=0.5, high=3., size=nsamples).astype(dtype) + 5.7 +\
        1j * np.random.uniform(low=0., high=6., size=nsamples).astype(dtype) + 21.2

    w = np.random.uniform(low=0., high=1e4, size=nsamples).astype(dtype)
    w /= w.sum()

    return x, y, w


########################################################
#                                                      #
#                      TESTS                           #
#                                                      #
########################################################
@pytest.mark.parametrize("size, real_type, tol, acc_lib",
                         [(1024, 'float32', 1.e-4, g_single),
                          (1024, 'float64', 1.e-13, g_double)],
                         ids=["SP", "DP"])
def test_uv_idx(size, real_type, tol, acc_lib):
    nsamples = 10
    maxuv = 1000.

    uv = pixel_coordinates(maxuv, size).astype(real_type)
    udat, vdat = create_sampling_points(nsamples, maxuv/4.8)
    assert len(udat) == nsamples
    assert len(vdat) == nsamples
    udat = udat.astype(real_type)
    vdat = vdat.astype(real_type)

    ui, vi = get_uv_idx_n(uv, uv, udat, vdat, len(uv))
    ui = ui.astype(real_type)
    vi = vi.astype(real_type)

    ui1, vi1 = acc_lib.get_uv_idx(size, maxuv/size, udat, vdat)

    np.testing.assert_allclose(ui1, ui, rtol=tol)
    np.testing.assert_allclose(vi1, vi, rtol=tol)


@pytest.mark.parametrize("size, real_type, complex_type, rtol, atol, acc_lib",
                         [(1024, 'float32', 'complex64',  1e-7,  1e-5, g_single),
                          (1024, 'float64', 'complex128', 1e-16, 1e-8, g_double)],
                         ids=["SP", "DP"])
def test_interpolate(size, real_type, complex_type, rtol, atol, acc_lib):
    nsamples = 10000
    maxuv = 1000.

    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=real_type)
    udat, vdat = create_sampling_points(nsamples, maxuv/2.2)
    # this factor has to be > than 2 because the matrix cover between -maxuv/2 to +maxuv/2,
    # therefore the sampling points have to be contained inside.

    udat = udat.astype(real_type)
    vdat = vdat.astype(real_type)

    # no rotation
    uv = pixel_coordinates(maxuv, size)
    uroti, vroti = get_uv_idx_n(uv, uv, udat, vdat, len(uv))

    uroti = uroti.astype(real_type)
    vroti = vroti.astype(real_type)

    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image))).astype(complex_type)

    # fortran
    ReInt = int_bilin(ft.real, uroti, vroti)
    ImInt = int_bilin(ft.imag, uroti, vroti)

    # gpu
    complexInt = acc_lib.interpolate(ft,
                                     uroti.astype(real_type),
                                     vroti.astype(real_type))

    np.testing.assert_allclose(ReInt, complexInt.real, rtol, atol)
    np.testing.assert_allclose(ImInt, complexInt.imag, rtol, atol)



@pytest.mark.parametrize("size, complex_type, rtol, atol, acc_lib",
                         [(1024, 'complex64', 1.e-5, 1e-3, g_single),
                          (1024, 'complex128', 1.e-16, 1e-8, g_double)],
                         ids=["SP", "DP"])
def test_FFT(size, complex_type, rtol, atol, acc_lib):

    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)

    ft = np.fft.fft2(reference_image)

    # create a copy of reference_image because galario makes in-place FFT
    ref_complex = reference_image.copy()
    acc_lib.fft2d(ref_complex)

    # print()
    # print(np.min(np.abs(ft.real)), np.max(np.abs(ft.real)))

    # some real parts can be very close to zero, so we need atol > 0!
    np.testing.assert_allclose(ft.real, ref_complex.real, rtol, atol)
    np.testing.assert_allclose(ft.imag, ref_complex.imag, rtol, atol)


@pytest.mark.parametrize("size, complex_type, rtol, atol, acc_lib",
                         [(1000, 'complex64',  1e-7, 1e-3, g_single),
                          (1000, 'complex128', 1.e-14, 1e-8, g_double)],
                         ids=["SP", "DP"])
def test_shift_fft_shift(size, complex_type, rtol, atol, acc_lib):

    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)
    cpu_shift_fft_shift = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image)))

    # create a copy of reference_image because galario makes in-place FFT
    ref_complex = reference_image.copy()
    acc_lib.fftshift_fft2d_fftshift(ref_complex)

    # see https://github.com/mtazzari/galario/issues/28
    # print()
    # print('%.15e' % cpu_shift_fft_shift.real[0,0])
    # print('%.15e' % ref_complex.real[0,0])
    # print('---')
    np.testing.assert_allclose(cpu_shift_fft_shift, ref_complex, rtol, atol)


@pytest.mark.parametrize("size, complex_type, tol, acc_lib",
                         [(1024, 'complex64', 1.e-8, g_single),
                          (1024, 'complex128', 1.e-16, g_double)],
                         ids=["SP", "DP"])
def test_shift(size, complex_type, tol, acc_lib):

    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)

    npshifted = np.fft.fftshift(reference_image)

    ref_complex = reference_image.copy()
    acc_lib.fftshift(ref_complex)

    np.testing.assert_allclose(npshifted, ref_complex, rtol=tol)


@pytest.mark.parametrize("real_type, complex_type, rtol, atol, acc_lib, pars",
                         [('float32', 'complex64',  1.e-7,  1e-5, g_single, par1),
                          ('float64', 'complex128', 1.e-16, 1e-15, g_double, par1),
                          ('float32', 'complex64',  1.e-3,  1e-5, g_single, par2),
                          ('float64', 'complex128', 1.e-16, 1e-14, g_double, par2),
                          ('float32', 'complex64',  1.e-7,  1e-5, g_single, par3),
                          ('float64', 'complex128', 1.e-16, 1e-15, g_double, par3)],
                         ids=["SP_par1", "DP_par1",
                              "SP_par2", "DP_par2",
                              "SP_par3", "DP_par3"])
def test_apply_phase(real_type, complex_type, rtol, atol, acc_lib, pars):

    wle_m = pars.get('wle_m', 0.003)
    x0_arcsec = pars.get('x0_arcsec', 0.4)
    y0_arcsec = pars.get('y0_arcsec', 10.)

    # generate the samples
    nsamples = 1000
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    # compute the matrix size and maxuv given the sampling points
    size, minuv, maxuv = matrix_size(udat, vdat)
    # print("size:{0}, minuv:{1}, maxuv:{2}".format(size, minuv, maxuv))

    # create reference image (complex)
    ref_complex = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)

    # compute the shift
    shifted_original_static = Fourier_shift_static(ref_complex, x0_arcsec, y0_arcsec, wle_m, maxuv)

    # GPU version
    factor = sec2rad/wle_m*maxuv
    acc_lib.apply_phase_2d(ref_complex, x0_arcsec * factor, y0_arcsec * factor)

    np.testing.assert_allclose(shifted_original_static, ref_complex, rtol, atol)


@pytest.mark.parametrize("real_type, complex_type, rtol, atol, acc_lib, pars",
                         [('float32', 'complex64',  1.e-7,  1e-5, g_single, par1),
                          ('float64', 'complex128', 1.e-16, 1e-14, g_double, par1),
                          ('float32', 'complex64',  1.e-3,  1e-5, g_single, par2),
                          ('float64', 'complex128', 1.e-16, 1e-14, g_double, par2),
                          ('float32', 'complex64',  1.e-7,  1e-5, g_single, par3),
                          ('float64', 'complex128', 1.e-16, 1e-15, g_double, par3)],
                         ids=["SP_par1", "DP_par1",
                              "SP_par2", "DP_par2",
                              "SP_par3", "DP_par3"])
def test_apply_phase_sampled(real_type, complex_type, rtol, atol, acc_lib, pars):

    x0_arcsec = pars.get('x0_arcsec', 0.4)
    y0_arcsec = pars.get('y0_arcsec', 10.)

    # generate the samples
    nsamples = 1000
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    # generate mock visibility values
    fint = np.zeros(nsamples, dtype=complex_type)
    fint.real = np.random.random(nsamples) * 10.
    fint.imag = np.random.random(nsamples) * 30.

    fint_numpy = Fourier_shift_array(udat, vdat, fint.copy(), x0_arcsec, y0_arcsec)

    acc_lib.apply_phase_sampled(x0_arcsec*sec2rad , y0_arcsec*sec2rad, udat, vdat, fint)

    np.testing.assert_allclose(fint_numpy.real, fint.real, rtol, atol)
    np.testing.assert_allclose(fint_numpy.imag, fint.imag, rtol, atol)





@pytest.mark.parametrize("nsamples, real_type, tol, acc_lib",
                         [(1000, 'float32', 1.e-6, g_single),
                          (1000, 'float64', 1.e-15, g_double)],
                         ids=["SP", "DP"])
def test_reduce_chi2(nsamples, real_type, tol, acc_lib):

    x, y, w = generate_random_vis(nsamples, real_type)
    chi2_ref = np.sum(((x.real - y.real) ** 2. + (x.imag - y.imag)**2.) * w)

    chi2_loc = acc_lib.reduce_chi2(x.real.copy(order='C'), x.imag.copy(order='C'), y.copy(), w)

    # print("Chi2_ref:{0}  Chi2_acc:{1}".format(chi2_ref, chi2_loc))
    # print("Absolute diff: {0}".format(chi2_loc-chi2_ref))
    # print("Relative diff: {0}".format((chi2_loc-chi2_ref)*2./(chi2_loc+chi2_ref)))

    np.testing.assert_allclose(chi2_ref, chi2_loc, rtol=tol)


# huge inaccuracy in single precision for larger images
@pytest.mark.parametrize("nsamples, real_type, complex_type, rtol, atol, acc_lib, pars",
                         [(100, 'float32', 'complex64',  1e-4,  1e-3, g_single, par1),
                          (1000, 'float64', 'complex128', 1e-14, 1e-10, g_double, par1)],
                         ids=["SP_par1", "DP_par1"])
def test_loss(nsamples, real_type, complex_type, rtol, atol, acc_lib, pars):
    # try to find out where precision is lost

    wle_m = pars.get('wle_m', 0.003)
    x0_arcsec = pars.get('x0_arcsec', 0.4)
    y0_arcsec = pars.get('y0_arcsec', 10.)

    # generate the samples
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    # compute the matrix size and maxuv
    size, minuv, maxuv = matrix_size(udat, vdat)
    uv = pixel_coordinates(maxuv, size)

    # create model complex image (it happens to have 0 imaginary part)
    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)
    ref_real = reference_image.real.copy()

    ###
    # shift - FFT - shift
    ###
    cpu_shift_fft_shift = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image)))

    shift_acc = reference_image.copy()
    acc_lib.fftshift_fft2d_fftshift(shift_acc)

    np.testing.assert_allclose(cpu_shift_fft_shift.real, shift_acc.real, rtol, atol)
    np.testing.assert_allclose(cpu_shift_fft_shift.imag, shift_acc.imag, rtol, atol)

    ###
    # phase
    ###
    factor = sec2rad/wle_m*maxuv
    fourier_shifted = Fourier_shift_static(cpu_shift_fft_shift, x0_arcsec, y0_arcsec, wle_m, maxuv)
    acc_lib.apply_phase_2d(shift_acc, x0_arcsec*factor, y0_arcsec*factor)

    # lose some absolute precision here  --> not anymore
    # atol *= 2
    np.testing.assert_allclose(fourier_shifted.real, shift_acc.real, rtol, atol)
    np.testing.assert_allclose(fourier_shifted.imag, shift_acc.imag, rtol, atol)
    # but continue with previous tolerance
    # atol /= 2

    ###
    # rotation indices
    ###
    uroti, vroti = get_uv_idx_n(uv, uv, udat, vdat, size)
    ui1, vi1 = acc_lib.get_uv_idx(size, maxuv/size, udat.astype(real_type), vdat.astype(real_type))

    np.testing.assert_allclose(ui1, uroti, rtol, atol)
    np.testing.assert_allclose(vi1, vroti, rtol, atol)


    ###
    # interpolation
    ###
    ReInt = int_bilin(fourier_shifted.real, uroti, vroti).astype(real_type)
    ImInt = int_bilin(fourier_shifted.imag, uroti, vroti).astype(real_type)
    complexInt = acc_lib.interpolate(shift_acc, uroti.astype(real_type), vroti.astype(real_type))

    np.testing.assert_allclose(ReInt, complexInt.real, rtol, atol)
    np.testing.assert_allclose(ImInt, complexInt.imag, rtol, atol)

    ###
    # now all steps in one function
    ###
    sampled = acc_lib.sample(ref_real, x0_arcsec, y0_arcsec,
                             maxuv/size/wle_m, udat/wle_m, vdat/wle_m)

    # a lot of precision lost. Why? --> not anymore
    # rtol = 1
    # atol = 0.5
    np.testing.assert_allclose(complexInt.real, sampled.real, rtol, atol)
    np.testing.assert_allclose(complexInt.imag, sampled.imag, rtol, atol)

    np.testing.assert_allclose(ReInt, sampled.real, rtol, atol)
    np.testing.assert_allclose(ImInt, sampled.imag, rtol, atol)

# single precision difference can be -1.152496e-01 vs 1.172152e+00 for large 1000x1000 images!!
@pytest.mark.parametrize("nsamples, real_type, complex_type, rtol, atol, acc_lib, pars",
                         [(100, 'float32', 'complex64',  1e-3,  1e-6, g_single, par1),
                          (1000, 'float64', 'complex128', 1e-12, 1e-11, g_double, par1),
                          (100, 'float32', 'complex64',  1e-3,  1e-4, g_single, par2), ## large x0, y0 induce larger errors
                          (1000, 'float64', 'complex128', 1e-12, 1e-10, g_double, par2),
                          (100, 'float32', 'complex64',  1e-3,  1e-5, g_single, par3),
                          (1000, 'float64', 'complex128', 1e-12, 1e-11, g_double, par3)],
                         ids=["SP_par1", "DP_par1",
                              "SP_par2", "DP_par2",
                              "SP_par3", "DP_par3"])
def test_sample(nsamples, real_type, complex_type, rtol, atol, acc_lib, pars):
    # go for fairly low precision when we add up many large numbers, we loose precision
    # TODO: perhaps implement the test with more realistic values of chi2 ~ 1

    wle_m = pars.get('wle_m', 0.003)
    x0_arcsec = pars.get('x0_arcsec', 0.4)
    y0_arcsec = pars.get('y0_arcsec', 10.)

    # generate the samples
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    # compute the matrix size and maxuv
    size, minuv, maxuv = matrix_size(udat, vdat)
    # print("size:{0}, minuv:{1}, maxuv:{2}".format(size, minuv, maxuv))
    uv = pixel_coordinates(maxuv, size).astype(real_type)

    # create model image (it happens to have 0 imaginary part)
    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)
    ref_real = reference_image.real.copy()

    # CPU version
    cpu_shift_fft_shift = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image)))
    fourier_shifted = Fourier_shift_static(cpu_shift_fft_shift, x0_arcsec, y0_arcsec, wle_m, maxuv)

    # compute interpolation and chi2
    uroti, vroti = get_uv_idx_n(uv, uv, udat, vdat, size)
    ReInt = int_bilin(fourier_shifted.real, uroti, vroti).astype(real_type)
    ImInt = int_bilin(fourier_shifted.imag, uroti, vroti).astype(real_type)

    # GPU
    sampled = acc_lib.sample(ref_real, x0_arcsec, y0_arcsec,
                             maxuv/size/wle_m, udat/wle_m, vdat/wle_m)

    np.testing.assert_allclose(ReInt, sampled.real, rtol, atol)
    np.testing.assert_allclose(ImInt, sampled.imag, rtol, atol)


@pytest.mark.parametrize("nsamples, real_type, complex_type, rtol, atol, acc_lib, pars",
                         [(1000, 'float32', 'complex64', 8.e-3, 8.e-3, g_single, par1),
                          (1000, 'float64', 'complex128', 1.e-14, 1.e-14, g_double, par1),
                          (1000, 'float32', 'complex64', 5.e-2, 8.e-3, g_single, par2),
                          (1000, 'float64', 'complex128', 1.e-10, 1.e-10, g_double, par2),
                          (1000, 'float32', 'complex64', 8.e-3, 8.e-3,g_single, par3),
                          (1000, 'float64', 'complex128', 1.e-14, 1.e-14, g_double, par3)],
                         ids=["SP_par1", "DP_par1",
                              "SP_par2", "DP_par2",
                              "SP_par3", "DP_par3"])
def test_chi2(nsamples, real_type, complex_type, rtol, atol, acc_lib, pars):
    # go for fairly low precision when we add up many large numbers, we loose precision
    # TODO: perhaps implement the test with more realistic values of chi2 ~ 1

    wle_m = pars.get('wle_m', 0.003)
    x0_arcsec = pars.get('x0_arcsec', 0.4)
    y0_arcsec = pars.get('y0_arcsec', 10.)

    # generate the samples
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)
    x, _, w = generate_random_vis(nsamples, real_type)

    # compute the matrix size and maxuv
    size, minuv, maxuv = matrix_size(udat, vdat)
    #print("size:{0}, minuv:{1}, maxuv:{2}".format(size, minuv, maxuv))
    uv = pixel_coordinates(maxuv, size).astype(real_type)

    # create model image (it happens to have 0 imaginary part)
    ref_complex = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)
    ref_real = ref_complex.real.copy()

    # CPU version
    cpu_shift_fft_shift = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ref_complex)))
    fourier_shifted = Fourier_shift_static(cpu_shift_fft_shift, x0_arcsec, y0_arcsec, wle_m, maxuv)

    # compute interpolation and chi2
    uroti, vroti = get_uv_idx_n(uv, uv, udat, vdat, size)
    ReInt = int_bilin(fourier_shifted.real, uroti, vroti).astype(real_type)
    ImInt = int_bilin(fourier_shifted.imag, uroti, vroti).astype(real_type)
    chi2_ref = np.sum(((ReInt - x.real) ** 2. + (ImInt - x.imag)**2.) * w)

    # GPU
    chi2_cuda = acc_lib.chi2(ref_real, x0_arcsec, y0_arcsec,
                             maxuv/size/wle_m, udat/wle_m, vdat/wle_m, x.real.copy(), x.imag.copy(), w)

    np.testing.assert_allclose(chi2_ref, chi2_cuda, rtol=rtol, atol=atol)


# a test case for profiling. Avoid python calls as much as possible.
def test_profile():
    nsamples = 512
    real_type = 'float64'
    complex_type = 'complex128'

    wle_m = par1.get('wle_m', 0.003)
    x0_arcsec = par1.get('x0_arcsec', 0.4)
    y0_arcsec = par1.get('y0_arcsec', 10.)

    # generate the samples
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)
    x, _, w = generate_random_vis(nsamples, real_type)

    # compute the matrix size and maxuv
    size, minuv, maxuv = matrix_size(udat, vdat, force_nx=4096)
    #print("size:{0}, minuv:{1}, maxuv:{2}".format(size, minuv, maxuv))
    uv = pixel_coordinates(maxuv, size).astype(real_type)

    # create model image (it happens to have 0 imaginary part)
    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=real_type)
    ref_complex = reference_image.copy()

    chi2_cuda = g_double.chi2(ref_complex, x0_arcsec, y0_arcsec,
                             maxuv/size/wle_m, udat/wle_m, vdat/wle_m, x.real.copy(), x.imag.copy(), w)
