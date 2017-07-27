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
    std::vector<dreal> data(nx*ny);
    galario_fft2d(nx, ny, &data[0]);
    galario_cleanup();

    return 0;
}
