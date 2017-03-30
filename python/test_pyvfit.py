import galario

import pyvfit
from pyvfit.imager import Imager
from pyvfit.constants import sec2rad
from pyvfit import vfit_ffun
from pyvfit.observations import ObsData
from pyvfit.star import Star
from pyvfit.uvtable import Uvtable

import numpy as np
import os
import pytest

# TODO make configurable
if galario.HAVE_CUDA:
    from galario import double_cuda as acc_lib
else:
    from galario import double as acc_lib

# TODO only tests double precision
ARR_2D_TYPE = 'complex128'
ARR_TYPE = 'float64'

# infer static dir from the pyvfit location. This only works if pyvfit is installed in developer mode
STATIC_DIR = os.path.join(os.path.split(pyvfit.__file__)[0], 'static')
TEST_REFERENCE_DIR = os.path.join(STATIC_DIR, 'test_reference')
TEST_SAMPLING_01 = os.path.join(TEST_REFERENCE_DIR, 'sampling_1024_100.npy')
TEST_FFT_01 = os.path.join(TEST_REFERENCE_DIR, 'fft_1024.npy')
TEST_FFT_02 = os.path.join(TEST_REFERENCE_DIR, 'fft_1024_noshift.npy')
TEST_OBSERVATIONS = os.path.join(TEST_REFERENCE_DIR, 'sample_obs.txt')
TEST_STAR = os.path.join(TEST_REFERENCE_DIR, 'test_star.dat')
TEST_MODEL = os.path.join(TEST_REFERENCE_DIR, 'test_twolayer.dat')
TEST_VISIBILITY_MAP = os.path.join(TEST_REFERENCE_DIR, 'test_visibility_map.npy')
TEST_VISIBILITY_MAP_0 = os.path.join(TEST_REFERENCE_DIR, 'test_visibility_map0.npy')


def create_reference_image(size=1024, kernel='gaussian', save=False):
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

    return reference_image


def create_sampling_points(nsamples, maxuv=1.):
    np.random.seed(42)

    # columns are non contiguous arrays => copy
    x = np.random.uniform(low=-maxuv, high=maxuv, size=(nsamples, 2))
    return x[:, 0].copy(), x[:, 1].copy()


def create_reference_sampling():

    size = 1024
    maxuv = 1000.
    nsamples = 100
    PA = 30./180.*np.pi  # deg

    reference_image = create_reference_image(size=size, kernel='gaussian')

    udat, vdat = create_sampling_points(nsamples, maxuv/4.8)

    uv = Imager.pixel_coordinates(maxuv, size)

    ReVis, ImVis = Imager.do_sampling(reference_image, udat, vdat, uv, size, PA)

    assert not np.any(np.isnan(ReVis))
    assert not np.any(np.isnan(ImVis))

    return np.hstack((ReVis, ImVis))


def save_reference_sampling():

    np.save(TEST_SAMPLING_01, create_reference_sampling())


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


def test_rotix():
    size = 1024
    nsamples = 10
    maxuv = 1000.

    uv = pyvfit.Imager.pixel_coordinates(maxuv, size).astype(ARR_TYPE)
    udat, vdat = create_sampling_points(nsamples, maxuv/4.8)
    assert len(udat) == nsamples
    assert len(vdat) == nsamples
    udat = udat.astype(ARR_TYPE)
    vdat = vdat.astype(ARR_TYPE)

    ui, vi = pyvfit.vfit_ffun.get_rotix_nf(uv, uv, udat, vdat, len(udat), size)
    ui = ui.astype(ARR_TYPE)
    vi = vi.astype(ARR_TYPE)

    #for i in xrange(len(udat)-1, -1, -1):
    #    print("{0}, {1}, {2}  (get_rotix_nf)".format(i, repr(ui[i]), repr(vi[i])))
    ui1, vi1 = acc_lib.acc_rotix(uv, udat, vdat)
    #for i in xrange(len(udat)-1, -1, -1):
    #    print("{0}, {1}, {2}  (cuda_rotix)".format(i, repr(ui1[i]), repr(vi1[i])))

    #for i in xrange(len(udat)-1, -1, -1):
    #    print("{0}, {1}, {2}  (cuda_rotix)".format(i, repr(ui1[i]-ui[i]), repr(vi1[i]-vi[i])))

    print(np.max((ui1-ui)*2./(ui1+ui)), np.max(ui1-ui))

    #tol = 1e-10  # for DOUBLE PRECISION
    tol = 1.e-4  # for SINGLE PRECISION
    np.testing.assert_allclose(ui1, ui, rtol=tol)
    np.testing.assert_allclose(vi1, vi, rtol=tol)


def test_sampling():
    reference = np.load(TEST_SAMPLING_01)

    np.testing.assert_allclose(create_reference_sampling(), reference, atol=1.e-16)



def test_interpolate():
    size = 1024
    nsamples = 10000
    maxuv = 1000.

    reference_image = create_reference_image(size=size, kernel='gaussian')
    udat, vdat = create_sampling_points(nsamples, maxuv/4.8)
    udat = udat.astype(ARR_TYPE)
    vdat = vdat.astype(ARR_TYPE)

    # no rotation
    uv = Imager.pixel_coordinates(maxuv, size)
    uroti, vroti = vfit_ffun.get_rotix_nf(uv, uv, udat, vdat, len(udat), size)
    uroti = uroti.astype(ARR_TYPE)
    vroti = vroti.astype(ARR_TYPE)

    fourier = Imager.numpy_FFT(reference_image).astype(ARR_2D_TYPE)
    # fortran
    ReInt = vfit_ffun.int_bilin_f(fourier.real, uroti, vroti, size, nsamples).astype(ARR_TYPE)
    ImInt = vfit_ffun.int_bilin_f(fourier.imag, uroti, vroti, size, nsamples).astype(ARR_TYPE)

    # gpu
    complexInt = acc_lib.acc_interpolate(fourier.astype(ARR_2D_TYPE), uroti.astype(ARR_TYPE), vroti.astype(ARR_TYPE))

    print(complexInt[0], ReInt[0], ImInt[0])
    print(np.max((ReInt-complexInt.real)*2./(ReInt+complexInt.real)), np.max(ReInt-complexInt.real))

    tol = 1e-3  # for SINGLE_PRECISION
    #tol = 1.e-16  # for DOUBLE_PRECISION
    np.testing.assert_allclose(ReInt, complexInt.real, rtol=tol)
    np.testing.assert_allclose(ImInt, complexInt.imag, rtol=tol)


def create_reference_shift_FFT_shift():
    size = 1024
    reference_image = create_reference_image(size=size, kernel='gaussian')
    shift_fft_shift = Imager.numpy_FFT(reference_image)

    assert not np.any(np.isnan(shift_fft_shift))

    return shift_fft_shift



def test_shift_fft_shift():
    size = 1024
    reference_image = create_reference_image(size=size, kernel='gaussian').astype(ARR_2D_TYPE)
    ref_complex = reference_image.copy().astype(ARR_2D_TYPE)
    cpu_shift_fft_shift = Imager.numpy_FFT(reference_image)
    acc_lib.acc_shift_fft_shift(ref_complex)

    tol = 0.1
    np.testing.assert_allclose(cpu_shift_fft_shift, ref_complex, rtol=tol)


def create_reference_FFT():
    size = 1024
    reference_image = create_reference_image(size=size, kernel='gaussian')
    ft = np.fft.fft2(reference_image)

    assert not np.any(np.isnan(ft))

    return ft



def test_FFT():

    size = 1024
    reference_image = create_reference_image(size=size, kernel='gaussian')

    ft = np.fft.fft2(reference_image)
    ref_complex = reference_image.astype(ARR_2D_TYPE)
    acc_lib.acc_fft(ref_complex)

    tol = 1.e-3  # for SINGLE PRECISION
    # tol = 1.e-16 # for DOUBLE PRECISION
    np.testing.assert_allclose(ft, ref_complex, atol=tol)



def test_shift():

    size = 1024
    reference_image = create_reference_image(size=size, kernel='gaussian')

    npshifted = np.fft.fftshift(reference_image)
    ref_complex = reference_image.astype('complex128')
    acc_lib.acc_shift(ref_complex)

    np.testing.assert_allclose(npshifted, ref_complex, atol=1.e-16)



def test_apply_phase():
    # generate the samples
    nsamples = 1000
    maxuv_generator = 1.e5
    udat, vdat = create_sampling_points(nsamples, maxuv_generator)
    # print(udat.min(), udat.max(), vdat.min(), vdat.max(), 4np.hypot(udat, vdat).max())

    # compute the matrix size and maxuv
    size, minuv, maxuv = Imager.matrix_size(udat, vdat)
    # print("size:{0}, minuv:{1}, maxuv:{2}".format(size, minuv, maxuv))

    # create reference complext image
    reference_image = create_reference_image(size=size, kernel='gaussian')
    ref_complex = reference_image.astype('complex128')

    # test values, can be whatever value
    wle = 0.3  # cm
    x0, y0 = 0.4, 10.

    # compute the original Fourier_shift
    imager = Imager(udat, vdat, wle, 1., 0.)
    shifted_original = imager.Fourier_shift(ref_complex, x0, y0)

    # compute the original Fourier_shift_static (static method)
    shifted_original_static = Imager.Fourier_shift_static(ref_complex, x0, y0, wle, maxuv)

    # compute the cuda version of Fourier shift
    factor = 2.*np.pi*sec2rad/wle*maxuv
    x0_cuda = x0 * factor
    y0_cuda = y0 * factor
    acc_lib.acc_apply_phase(ref_complex, x0_cuda, y0_cuda)

    np.testing.assert_allclose(shifted_original_static, ref_complex, atol=1.e-16)
    np.testing.assert_allclose(shifted_original, ref_complex, atol=1.e-16)



def test_chi2():
    obsData = ObsData('', par_obs=dict(data_filenames=[TEST_OBSERVATIONS], wle_mm=[0.88], weight_corr=[1.], disk_ffcontr=[0.]))
    uvtable = obsData.uvtables[0]

    # make up model predictions by modifying the observed values
    pred = np.zeros((len(uvtable.re),), dtype=np.dtype("complex128"))
    pred.real[:] = np.copy(uvtable.re)
    pred.imag[:] = np.copy(uvtable.im)

    pred.real *= 1.5
    pred.imag *= 0.89

    chi2_ref = uvtable.compute_chisquare(pred.real, pred.imag)
    chi2_loc = acc_lib.acc_chi2(uvtable.re, uvtable.im, pred, uvtable.w)

    print("Chi2_ref:{0}  Chi2_acc:{1}".format(chi2_ref, chi2_loc))
    print("Absolute diff: {0}".format(chi2_loc-chi2_ref))
    print("Relative diff: {0}".format((chi2_loc-chi2_ref)*2./(chi2_loc+chi2_ref)))

    if galario.HAVE_CUDA:
        rtol = 1.e-15
    else:
        rtol = 1.e-14

    np.testing.assert_allclose(chi2_ref, chi2_loc, rtol=rtol)



def test_doeverything():
    # Let us define "everything" as: shift, Fourier, shift, apply_phase, rotix, interpolate, compute chisquare
    # read observational data
    obsData = ObsData('', par_obs=dict(data_filenames=[TEST_OBSERVATIONS], wle_mm=[0.88], weight_corr=[1.], disk_ffcontr=[0.]))
    uvtable = obsData.uvtables[0]

    # take the samples from the observations
    udat, vdat = uvtable.u, uvtable.v
    nsamples = len(udat)
    # print(udat.min(), udat.max(), vdat.min(), vdat.max(), 4np.hypot(udat, vdat).max())

    # compute the matrix size and maxuv
    size, minuv, maxuv = Imager.matrix_size(udat, vdat)
    print("size:{0}, minuv:{1}, maxuv:{2}".format(size, minuv, maxuv))
    uv = Imager.pixel_coordinates(maxuv, size)
    wle = uvtable.wle  # cm

    # create model complex image (it happens to have 0 imaginary part)
    reference_image = create_reference_image(size=size, kernel='gaussian')
    ref_complex = reference_image.astype('complex128')

    # test values, can be whatever value
    x0, y0 = 2.5, 3.7

    # execute the CPU version of everything
    cpu_shift_fft_shift = Imager.numpy_FFT(reference_image)
    # compute the original Fourier_shift
    imager = Imager(udat, vdat, wle, 1., 0.)
    fourier_shifted = imager.Fourier_shift(cpu_shift_fft_shift, x0, y0)
    # OR compute the original Fourier_shift_static (static method)
    #shifted_original_static = Imager.Fourier_shift_static(ref_complex, x0, y0, wle, maxuv)
    # compute interpolation indices
    uroti, vroti = vfit_ffun.get_rotix_nf(uv, uv, udat, vdat, len(udat), size)
    ReInt = vfit_ffun.int_bilin_f(fourier_shifted.real, uroti, vroti, size, nsamples)
    ImInt = vfit_ffun.int_bilin_f(fourier_shifted.imag, uroti, vroti, size, nsamples)
    chi2_ref = uvtable.compute_chisquare(ReInt, ImInt)

    # execute the GPU version of everything
    factor = 2.*np.pi*sec2rad/wle*maxuv
    x0_cuda = x0 * factor
    y0_cuda = y0 * factor

    # gpu
    rank = 0  # MPI rank
    chi2_cuda = acc_lib.acc_everything(ref_complex, x0_cuda, y0_cuda, uv, udat, vdat, uvtable.re, uvtable.im, uvtable.w, rank)

    print(chi2_ref, chi2_cuda)

    # go for fairly low precision when we add up many large numbers, we loose precision
    np.testing.assert_allclose(chi2_ref, chi2_cuda, rtol=1e-10)


# note that the rotated indices are computed different by get_rotix_nf and cuda_rotix (see comment on test_rotix())
# this reflects in absolute discrepancy ~1.e-5.
# to be assessed: if this error on the interpolated points is acceptable or not. Now we assume it is
@pytest.mark.skipif("cuda_rotix_interpolate" not in dir(acc_lib), reason="requires cuda and the cuda_rotix_interpolate function")
def test_rotix_interpolate():
    size = 1024
    nsamples = 10
    maxuv = 1000.

    uv = Imager.pixel_coordinates(maxuv, size)
    udat, vdat = create_sampling_points(nsamples, maxuv/4.8)

    reference_image = create_reference_image(size=size, kernel='gaussian')
    fourier = Imager.numpy_FFT(reference_image)

    # no rotation of udat, vdat

    # fortran
    uroti, vroti = vfit_ffun.get_rotix_nf(uv, uv, udat, vdat, len(udat), size)
    #uroti, vroti = acc_lib.acc_rotix(uv, udat, vdat)
    ReInt = vfit_ffun.int_bilin_f(fourier.real, uroti, vroti, size, nsamples)
    ImInt = vfit_ffun.int_bilin_f(fourier.imag, uroti, vroti, size, nsamples)

    # gpu
    complexInt = acc_lib.acc_rotix_interpolate(uv, udat, vdat, fourier)

    tol = 1e-6
    #for i in xrange(len(udat)-1, -1, -1):
    #    print("{0}, {1}, {2}  (py)".format(i, repr(uroti[i]), repr(vroti[i])))


    # print(ImInt/complexInt.imag)
    np.testing.assert_allclose(ReInt, complexInt.real, atol=tol)
    np.testing.assert_allclose(ImInt, complexInt.imag, atol=tol)


@pytest.mark.skipif("cuda_shift_fft_shift_apply_phase" not in dir(acc_lib), reason="requires cuda and the cuda_shift_fft_shift_apply_phase function")
def test_shift_fft_shift_apply_phase():
    # generate the samples
    nsamples = 1000
    maxuv_generator = 1.e5
    udat, vdat = create_sampling_points(nsamples, maxuv_generator)
    # print(udat.min(), udat.max(), vdat.min(), vdat.max(), 4np.hypot(udat, vdat).max())

    # compute the matrix size and maxuv
    size, minuv, maxuv = Imager.matrix_size(udat, vdat)
    print("size:{0}, minuv:{1}, maxuv:{2}".format(size, minuv, maxuv))

    # create reference complext image
    reference_image = create_reference_image(size=size, kernel='gaussian')
    ref_complex = reference_image.astype('complex128')

    # test values, can be whatever value
    wle = 0.3  # cm
    x0, y0 = 0.4, 10.

    # execute the CPU version of shift, Fourier, shift, apply_phase
    cpu_shift_fft_shift = Imager.numpy_FFT(reference_image)
    # compute the original Fourier_shift
    imager = Imager(udat, vdat, wle, 1., 0.)
    shifted_original = imager.Fourier_shift(cpu_shift_fft_shift, x0, y0)
    # OR compute the original Fourier_shift_static (static method)
    #shifted_original_static = Imager.Fourier_shift_static(ref_complex, x0, y0, wle, maxuv)

    # execute the GPU version of shift, Fourier, shift, apply_phase
    factor = 2.*np.pi*sec2rad/wle*maxuv
    x0_cuda = x0 * factor
    y0_cuda = y0 * factor
    acc_lib.acc_shift_fft_shift_apply_phase(ref_complex, x0_cuda, y0_cuda)

    np.testing.assert_allclose(shifted_original, ref_complex, atol=1.e-16)

@pytest.mark.skipif(True, reason="currently broken, issue #39")
def test_visibility_map():
    from py2layer import TwoLayer_g_7K

    # read observational data
    obsData = ObsData('', par_obs=dict(data_filenames=[TEST_OBSERVATIONS], wle_mm=[0.88], weight_corr=[1.], disk_ffcontr=[0.]))
    uvtable = obsData.uvtables[0]
    star = Star('test_star', TEST_STAR)

    # test values, can be whatever value
    # x0, y0 = 2.5, 3.7

    model = TwoLayer_g_7K(star, TEST_MODEL)
    model.compute_grids()
    result, exit_code = model(1., 10., 100., 0.1, 0., 30.)

    imager = Imager(uvtable.u, uvtable.v, uvtable.wle, star.dist.cm, 0.)

    inc = 0.*np.pi/180.
    PA = 0.*np.pi/180.
    delta_alpha = -0.38
    delta_delta = 0.52
    intensmap = imager.intensity_map(result[0, :], model.gridrad, inc)
    vrm, vim = imager.visibility_map(intensmap, delta_alpha, delta_delta, PA, uvtable.u, uvtable.v )

    # uvtable.set_model(vrm, vim)
    # uvtable.export_model('./model.txt')
    # np.save(TEST_VISIBILITY_MAP, [vrm, vim])  # inc=30, PA=130
    # np.save(TEST_VISIBILITY_MAP_0, [vrm, vim])  # inc=0, PA=0


    reference = np.load(TEST_VISIBILITY_MAP_0)

    print(vrm/reference[0])
    print(np.mean(vrm/reference[0]), np.max(vrm/reference[0]))
    np.testing.assert_allclose(vrm, reference[0], rtol=1.e-8)
    np.testing.assert_allclose(vim, reference[1], rtol=1.e-8)

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
