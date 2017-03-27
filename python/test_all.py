import pygalario
# from pygalario import HAVE_CUDA
import pyvfit
from pyvfit.cuda import pyvfit_libcpu as acc_lib
import numpy as np
import os
import pytest

# TODO only test double precision
ARR_2D_TYPE = 'complex128'
ARR_TYPE = 'float64'

# infer static dir from the pyvfit location. This only works if pyvfit is installed in developer mode
STATIC_DIR = os.path.join(os.path.split(pyvfit.__file__)[0], 'static')
# STATIC_DIR = "${PYVFIT_STATIC_DIR}"
# TEST_REFERENCE_DIR = ${PYVFIT_REFERENCE_DIR}
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


# @pytest.mark.skipif(not HAVE_CUDA, reason="requires cuda")
def test_rotix():
    # from .imager import Imager
    # from . import vfit_ffun

    size = 1024
    nsamples = 10
    maxuv = 1000.

    uv = pyvfit.Imager.pixel_coordinates(maxuv, size).astype(ARR_TYPE)
    udat, vdat = create_sampling_points(nsamples, maxuv/4.8)
    assert len(udat) == nsamples
    assert len(vdat) == nsamples
    udat = udat.astype(ARR_TYPE)
    vdat = vdat.astype(ARR_TYPE)

    # ui, vi = pyvfit.vfit_ffun.get_rotix_nf(uv, uv, udat, vdat, len(udat), size)
    # ui = ui.astype(ARR_TYPE)
    # vi = vi.astype(ARR_TYPE)
    ui, vi = pygalario.double.acc_rotix(uv, udat, vdat)

    ui1, vi1 = acc_lib.acc_rotix(uv, udat, vdat)

    tol = 1.e-16  # for SINGLE PRECISION
    np.testing.assert_allclose(ui1, ui, rtol=tol)
    np.testing.assert_allclose(vi1, vi, rtol=tol)
