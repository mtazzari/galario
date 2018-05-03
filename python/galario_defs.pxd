include "galario_config.pxi"

cdef extern from "galario_py.h" namespace "galario":
    # Main user functions
    void _sample_profile(int nr, void* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis) except +
    void _sample_image(int nx, int ny, void* image, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis) except +
    void _chi2_profile(int nr, void* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis_obs_re, void* vis_obs_im, void* vis_obs_w, dreal* chi2) except +
    void _chi2_image(int nx, int ny, void* image, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis_obs_re, void* vis_obs_im, void* vis_obs_w, dreal* chi2) except +
    void _sweep(int nr, void* intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, void* image) except +
    void _uv_rotate(dreal PA, dreal dRA, dreal dDec, void* dRArot, void* dDecrot, int nd, void* u, void* v, void* urot, void* vrot) except +

    # Interface for the experts
    void* _copy_input(int nx, int ny, void* realimage) except +
    void* _fft2d(int nx, int ny, void* image) except +
    void _fftshift(int nx, int ny, void* image) except +
    void _fftshift_axis0(int nx, int ny, void* image) except +
    void _interpolate(int nx, int ncol, void* image, int nd, void* u, void* v, dreal duv, void* vis) except +
    void _apply_phase_sampled(dreal dRA, dreal dDec, int nd, void* u, void* v, void* vis) except +
    void _reduce_chi2(int nd, void* vis_obs_re, void* vis_obs_im, void* vis, void* vis_obs_w, dreal* chi2) except +

cdef extern from "galario.h" namespace "galario":
    void init();
    void cleanup();
    int  threads(int num);
    void galario_free(void*);
    void use_gpu(int device_id)
    int  ngpus()