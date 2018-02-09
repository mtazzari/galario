/*
 * A simple test to make sure we can include, link, and run galario from pure C++.
 */
#include "galario.h"

#include <cassert>
#include <complex>
#include <vector>

int main()
{
    galario_init();
    constexpr int nx = 128;
    constexpr int ny = 128;

    // vector initializes to 0
    std::vector<dreal> realdata(nx*ny);
    dcomplex* res = galario_copy_input(nx, ny, &realdata[0]);

    // These checks compute garbage but we want to check if they compile and link.
    // Link errors could come from incompatible const modifiers.
    int n = 4;
    dreal* rp = &realdata[0];
    dcomplex* cp = res;
    dreal r = realdata[0];

    // TODO all checks involving interpolate can segfault when accessing invalid adresses
    // due to invalid inputs in u, v, and duv.

    // galario_sample_profile(n, rp, r, r, r, n, r, r, r, r, r, n, rp, rp, cp);
    // galario_sample_image(n, n, rp, r, r, r, r, n, rp, rp, cp);
    // galario_chi2_profile(n, rp, r, r, r, n, r, r, r, r, r, n, rp, rp, rp, rp, rp, rp);
    // galario_chi2_image(n, n, rp, r, r, r, r, n, rp, rp, rp, rp, rp, rp);
    galario_sweep(n, rp, r, r, n, r, r, cp);
    galario_uv_rotate(r, r, r, rp, rp, n, rp, rp ,rp, rp);

    galario_fft2d(nx, ny, res);
    galario_fftshift(n, n, cp);
    galario_fftshift_axis0(n, n, cp);
    // galario_interpolate(n, n, cp, n, rp, rp, r, cp);
    galario_apply_phase_sampled(r, r, n, rp, rp, cp);

    auto ncomplex = nx*(ny/2+1);
    std::vector<dcomplex> fint(res, res + ncomplex);
    auto chi2 = galario_reduce_chi2(300, &realdata[0], &realdata[0], res, &realdata[0]);
    (void)chi2; // avoid unused variable warning

    // copy input before galario_reduce_chi2 to see if it is modified inadvertently
    for (auto i = 0; i < ncomplex; ++i) {
        assert(fint[i] == res[i]);
    }

    galario_free(res);
    galario_cleanup();

    return 0;
}
