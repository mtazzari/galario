/*
 * A simple test to make sure we can include, link, and run galario from pure C++.
 */
#include "galario.h"

#include <complex>
#include <vector>

int main()
{
    galario_init();
    constexpr int nx = 128;
    constexpr int ny = 128;
    std::vector<dreal> realdata(nx*ny);
    dcomplex* res = galario_copy_input(nx, ny, &realdata[0]);
    galario_fft2d(nx, ny, res);
    galario_free(res);
    galario_cleanup();

    return 0;
}
