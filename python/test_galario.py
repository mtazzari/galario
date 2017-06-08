import numpy as np
import os
import pytest

import galario

# TODO make configurable pass cmd option
if galario.HAVE_CUDA:  # and option py.test has --gpu
    from galario import double_cuda as g_double
    from galario import single_cuda as g_single
else:
    from galario import double as g_double
    from galario import single as g_single

# PARAMETERS FOR MULTIPLE TEST EXECUTIONS
par1 = {'wle_m': 0.003, 'x0_arcsec': 0.4, 'y0_arcsec': 4.}
par2 = {'wle_m': 0.00088, 'x0_arcsec': -3.5, 'y0_arcsec': 7.2}
par3 = {'wle_m': 0.00088, 'x0_arcsec': 0., 'y0_arcsec': 0.}

sec2rad = 4.848136811e-06  # from arcsec to radians

# TO BE REMOVED

# tests still to fix
#  - test_visibility_map
#  - test_rotix_interpolate

# END TO BE REMOVED


# use last gpu if available. Check `watch -n 0.1 nvidia-smi` to see which gpu is
# used during test execution.
ngpus = g_double.ngpus()
g_double.use_gpu(max(0, ngpus-1))

########################################################
#                                                      #
#                  REFERENCE FUNCTIONS                 #
#                                                      #
########################################################
def create_reference_image(size=1024, kernel='gaussian', save=False, dtype='float64'):
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
        plt.savefig(os.path.join(TEST_REFERENCE_DIR, 'reference_image.pdf'))

    return reference_image.astype(dtype)


def create_sampling_points(nsamples, maxuv=1., dtype='float64'):
    np.random.seed(42)

    # columns are non contiguous arrays => copy
    x = np.random.uniform(low=-maxuv, high=maxuv, size=(nsamples, 2))
    return x[:, 0].astype(dtype), x[:, 1].astype(dtype)


def create_reference_sampling():

    size = 1024
    maxuv = 1000.
    nsamples = 100
    PA = 30./180.*np.pi  # deg

    reference_image = create_reference_image(size=size, kernel='gaussian')

    udat, vdat = create_sampling_points(nsamples, maxuv/4.8)

    uv = pixel_coordinates(maxuv, size)

    ReVis, ImVis = Imager.do_sampling(reference_image, udat, vdat, uv, size, PA)

    assert not np.any(np.isnan(ReVis))
    assert not np.any(np.isnan(ImVis))

    return np.hstack((ReVis, ImVis))


def rotix(udat, vdat, uv):
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

    return (np.linspace(0., nx-1, nx) - nx/2.) * maxuv/(nx-1)


def get_rotix_n(ux, vx, ur, vr, size):

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
        ii = int(x[i])
        jj = int(y[i])
        dfj = f[ii + 1, jj] - f[ii, jj]           # x
        dfj1 = f[ii + 1, jj + 1] - f[ii, jj + 1]  # y
        fix = f[ii, jj] + dfj * (x[i] - int(x[i]))
        fix1 = f[ii + 1, jj] + dfj1 * (x[i] - int(x[i]))
        fint[i] = fix + (fix1 - fix) * (y[i] - int(y[i]))

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
    uu, vv = np.meshgrid(spatial_freq*y0, spatial_freq*x0)
    uv_grid = uu+vv
    cos_theta = np.cos(uv_grid)
    sin_theta = -np.sin(uv_grid)

    # apply the phase change
    re_ft_c, im_ft_c = ft_centered.real, ft_centered.imag
    re_v_shifted = re_ft_c*cos_theta - im_ft_c*sin_theta
    imag_v_shifted = im_ft_c*cos_theta + re_ft_c*sin_theta

    v_shifted = re_v_shifted+1j*imag_v_shifted

    return v_shifted

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
def test_rotix(size, real_type, tol, acc_lib):
    nsamples = 10
    maxuv = 1000.

    uv = pixel_coordinates(maxuv, size).astype(real_type)
    udat, vdat = create_sampling_points(nsamples, maxuv/4.8)
    assert len(udat) == nsamples
    assert len(vdat) == nsamples
    udat = udat.astype(real_type)
    vdat = vdat.astype(real_type)

    ui, vi = get_rotix_n(uv, uv, udat, vdat, size)
    ui = ui.astype(real_type)
    vi = vi.astype(real_type)

    ui1, vi1 = acc_lib.acc_rotix(uv, udat, vdat)

    np.testing.assert_allclose(ui1, ui, rtol=tol)
    np.testing.assert_allclose(vi1, vi, rtol=tol)


@pytest.mark.parametrize("size, real_type, complex_type, tol, acc_lib",
                         [(1024, 'float32', 'complex64', 1.e-2, g_single),
                          (1024, 'float64', 'complex128', 1.e-16, g_double)],
                         ids=["SP", "DP"])
def test_interpolate(size, real_type, complex_type, tol, acc_lib):
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
    uroti, vroti = get_rotix_n(uv, uv, udat, vdat, size)

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

    np.testing.assert_allclose(ReInt, complexInt.real, rtol=tol)
    np.testing.assert_allclose(ImInt, complexInt.imag, rtol=tol)



@pytest.mark.parametrize("size, complex_type, tol, acc_lib",
                         [(1024, 'complex64', 1.e-3, g_single),
                          (1024, 'complex128', 1.e-16, g_double)],
                         ids=["SP", "DP"])
def test_FFT(size, complex_type, tol, acc_lib):

    reference_image = create_reference_image(size=size, kernel='gaussian')

    ft = np.fft.fft2(reference_image)

    # create a copy of reference_image because galario makes in-place FFT
    ref_complex = reference_image.astype(complex_type)
    acc_lib.fft2d(ref_complex)

    # tol = 1.e-3  # for SINGLE PRECISION
    # tol = 1.e-16 # for DOUBLE PRECISION
    np.testing.assert_allclose(ft, ref_complex, atol=tol)


@pytest.mark.parametrize("size, complex_type, tol, acc_lib",
                         [(1024, 'complex64', 0.2, g_single),
                          (1024, 'complex128', 1.e-9, g_double)],
                         ids=["SP", "DP"])
def test_shift_fft_shift(size, complex_type, tol, acc_lib):

    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)
    cpu_shift_fft_shift = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image)))

    # create a copy of reference_image because galario makes in-place FFT
    ref_complex = reference_image.copy().astype(complex_type)
    acc_lib.fftshift_fft2d_fftshift(ref_complex)

    np.testing.assert_allclose(cpu_shift_fft_shift, ref_complex, rtol=tol)


@pytest.mark.parametrize("size, complex_type, tol, acc_lib",
                         [(1024, 'complex64', 1.e-7, g_single),
                          (1024, 'complex128', 1.e-16, g_double)],
                         ids=["SP", "DP"])
def test_shift(size, complex_type, tol, acc_lib):

    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)

    npshifted = np.fft.fftshift(reference_image)

    ref_complex = reference_image.copy()
    acc_lib.fftshift(ref_complex)

    np.testing.assert_allclose(npshifted, ref_complex, rtol=tol)


@pytest.mark.parametrize("real_type, complex_type, tol, acc_lib, pars",
                         [('float32', 'complex64', 1.e-4, g_single, par1),
                          ('float64', 'complex128', 1.e-11, g_double, par1),
                          ('float32', 'complex64', 1.e-3, g_single, par2),
                          ('float64', 'complex128', 1.e-11, g_double, par2),
                          ('float32', 'complex64', 1.e-3, g_single, par3),
                          ('float64', 'complex128', 1.e-11, g_double, par3)],
                         ids=["SP_par1", "DP_par1",
                              "SP_par2", "DP_par2",
                              "SP_par3", "DP_par3"])
def test_apply_phase(real_type, complex_type, tol, acc_lib, pars):

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
    factor = 2.*np.pi*sec2rad/wle_m*maxuv
    acc_lib.apply_phase_2d(ref_complex, x0_arcsec * factor, y0_arcsec * factor)

    np.testing.assert_allclose(shifted_original_static, ref_complex, rtol=tol)


@pytest.mark.parametrize("nsamples, real_type, tol, acc_lib",
                         [(1000, 'float32', 1.e-6, g_single),
                          (1000, 'float64', 1.e-14, g_double)],
                         ids=["SP", "DP"])
def test_reduce_chi2(nsamples, real_type, tol, acc_lib):

    x, y, w = generate_random_vis(nsamples, real_type)
    chi2_ref = np.sum(((x.real - y.real) ** 2. + (x.imag - y.imag)**2.) * w)

    chi2_loc = acc_lib.reduce_chi2(x.real.copy(order='C'), x.imag.copy(order='C'), y.copy(), w)

    # print("Chi2_ref:{0}  Chi2_acc:{1}".format(chi2_ref, chi2_loc))
    # print("Absolute diff: {0}".format(chi2_loc-chi2_ref))
    # print("Relative diff: {0}".format((chi2_loc-chi2_ref)*2./(chi2_loc+chi2_ref)))

    np.testing.assert_allclose(chi2_ref, chi2_loc, rtol=tol)


@pytest.mark.parametrize("nsamples, real_type, complex_type, tol, acc_lib, pars",
                         [(1000, 'float32', 'complex64', 8.e-3, g_single, par1),
                          (1000, 'float64', 'complex128', 1.e-14, g_double, par1),
                          (1000, 'float32', 'complex64', 5.e-2, g_single, par2),
                          (1000, 'float64', 'complex128', 1.e-14, g_double, par2),
                          (1000, 'float32', 'complex64', 8.e-3, g_single, par3),
                          (1000, 'float64', 'complex128', 1.e-14, g_double, par3)],
                         ids=["SP_par1", "DP_par1",
                              "SP_par2", "DP_par2",
                              "SP_par3", "DP_par3"])
def test_chi2(nsamples, real_type, complex_type, tol, acc_lib, pars):
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
    print("size:{0}, minuv:{1}, maxuv:{2}".format(size, minuv, maxuv))
    uv = pixel_coordinates(maxuv, size).astype(real_type)

    # create model complex image (it happens to have 0 imaginary part)
    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)
    ref_complex = reference_image.astype(complex_type)

    # CPU version
    cpu_shift_fft_shift = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image)))
    fourier_shifted = Fourier_shift_static(cpu_shift_fft_shift, x0_arcsec, y0_arcsec, wle_m, maxuv)

    # compute interpolation and chi2
    uroti, vroti = get_rotix_n(uv, uv, udat, vdat, size)
    ReInt = int_bilin(fourier_shifted.real, uroti, vroti).astype(real_type)
    ImInt = int_bilin(fourier_shifted.imag, uroti, vroti).astype(real_type)
    chi2_ref = np.sum(((ReInt - x.real) ** 2. + (ImInt - x.imag)**2.) * w)

    # to be tested
    # shift_array = Imager.Fourier_shift_array(u, v, Re, Im, x0, y0, wle)

    # GPU
    factor = 2.*np.pi*sec2rad/wle_m*maxuv

    chi2_cuda = acc_lib.chi2(ref_complex, x0_arcsec * factor, y0_arcsec * factor,
                             uv, udat, vdat, x.real.copy(), x.imag.copy(), w)

    np.testing.assert_allclose(chi2_ref, chi2_cuda, rtol=tol, atol=0.1)


# note that the rotated indices are computed different by get_rotix_nf and cuda_rotix (see comment on test_rotix())
# this reflects in absolute discrepancy ~1.e-5.
# to be assessed: if this error on the interpolated points is acceptable or not. Now we assume it is
@pytest.mark.skipif(True, reason="requires cuda and the cuda_rotix_interpolate function")
def test_rotix_interpolate():
    size = 1024
    nsamples = 10
    maxuv = 1000.

    uv = pixel_coordinates(maxuv, size)
    udat, vdat = create_sampling_points(nsamples, maxuv/4.8)

    reference_image = create_reference_image(size=size, kernel='gaussian')
    fourier = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image)))

    # no rotation of udat, vdat

    # fortran
    uroti, vroti = get_rotix_n(uv, uv, udat, vdat, size)
    #uroti, vroti = acc_lib.acc_rotix(uv, udat, vdat)
    ReInt = int_bilin(fourier.real, uroti, vroti)
    ImInt = int_bilin(fourier.imag, uroti, vroti)

    # gpu
    complexInt = acc_lib.acc_rotix_interpolate(uv, udat, vdat, fourier)

    tol = 1e-6
    #for i in range(len(udat)-1, -1, -1):
    #    print("{0}, {1}, {2}  (py)".format(i, repr(uroti[i]), repr(vroti[i])))


    # print(ImInt/complexInt.imag)
    np.testing.assert_allclose(ReInt, complexInt.real, atol=tol)
    np.testing.assert_allclose(ImInt, complexInt.imag, atol=tol)


@pytest.mark.skipif(True, reason="currently broken, issue #39")
def test_visibility_map():
    from py2layer import TwoLayer_g_7K

    # read observational data
    obsData = ObsData('', par_obs=dict(data_filenames=[TEST_OBSERVATIONS], wle_mm=[0.88], weight_corr=[1.], disk_ffcontr=[0.]))
    uvtable = obsData.uvtables[0]
    star = Star('test_star', TEST_STAR)

    u, v, w = uvtable.u, uvtable.v, uvtable.w
    # test values, can be whatever value
    # x0, y0 = 2.5, 3.7

    model = TwoLayer_g_7K(star, TEST_MODEL)
    model.compute_grids()

    imager = Imager(uvtable.u, uvtable.v, uvtable.wle, star.dist.cm, 0.)

    # mock disk
    gridrad = np.logspace(np.log10(0.1), np.log10(600.), 500)
    brightness = np.exp(-(gridrad / 100.) ** 2. / 2.)*1.e-11

    inc = 30.*np.pi/180.
    PA = 25.*np.pi/180.
    delta_alpha = -0.38
    delta_delta = 0.52
    intensmap = imager.intensity_map(brightness, gridrad, inc)
    vrm, vim = imager.visibility_map(intensmap, delta_alpha, delta_delta, PA, u, v)

    uvtable.set_model(vrm, vim)
    uvtable.export_model("mock_observations.txt")
    np.save("mock_vis_sampled.npy", [u, v, vrm, vim, w])
    np.savetxt("mock_vis_sampled.txt", [vrm, vim])


    # BUT: THIS FUNCTION DOES NOT TEST GALARIO!!!?!?!?!?!?

    # uvtable.set_model(vrm, vim)
    # uvtable.export_model('./model.txt')
    # np.save(TEST_VISIBILITY_MAP, [vrm, vim])  # inc=30, PA=130
    # np.save(TEST_VISIBILITY_MAP_0, [vrm, vim])  # inc=0, PA=0


    reference = np.load(TEST_VISIBILITY_MAP_0)

    print(vrm/reference[0])
    print(np.mean(vrm/reference[0]), np.max(vrm/reference[0]))
    np.testing.assert_allclose(vrm, reference[0], rtol=1.e-6)
    np.testing.assert_allclose(vim, reference[1], rtol=1.e-6)

    # execute the CPU version of everything
    # cpu_shift_fft_shift = Imager.numpy_FFT(reference_image)
    # # compute the original Fourier_shift
    # fourier_shifted = imager.Fourier_shift(cpu_shift_fft_shift, x0, y0)
    # # OR compute the original Fourier_shift_static (static method)
    # #shifted_original_static = Imager.Fourier_shift_static(ref_complex, x0, y0, wle, maxuv)
    # # compute interpolation indices
    # uroti, vroti = vfit_ffun.get_rotix_nf(uv, uv, udat, vdat, len(udat), size)
    # ReInt = vfit_ffun.int_bilin_f(fourier_shifted.real, uroti, vroti, size, nsamples)
    # ImInt = vfit_ffun.int_bilin_f(fourier_shifted.imag, uroti, vroti, size, nsamples)
    # chi2_ref = uvtable.compute_chisquare(ReInt, ImInt)

    # go for fairly low precision when we add up many large numbers, we loose precision
    # np.testing.assert_allclose(chi2_ref, chi2_cuda, rtol=1e-8)
