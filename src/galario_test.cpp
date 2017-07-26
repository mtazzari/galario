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
    std::vector<dreal> data(nx*nx);
    galario_fft2d(nx, &data[0]);
    galario_cleanup();

    return 0;
}
