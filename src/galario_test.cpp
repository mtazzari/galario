/******************************************************************************
* This file is part of GALARIO:                                               *
* Gpu Accelerated Library for Analysing Radio Interferometer Observations     *
*                                                                             *
* Copyright (C) 2017-2018, Marco Tazzari, Frederik Beaujean, Leonardo Testi.  *
*                                                                             *
* This program is free software: you can redistribute it and/or modify        *
* it under the terms of the Lesser GNU General Public License as published by *
* the Free Software Foundation, either version 3 of the License, or           *
* (at your option) any later version.                                         *
*                                                                             *
* This program is distributed in the hope that it will be useful,             *
* but WITHOUT ANY WARRANTY; without even the implied warranty of              *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                        *
*                                                                             *
* For more details see the LICENSE file.                                      *
* For documentation see https://mtazzari.github.io/galario/                   *
******************************************************************************/

/*
 * A simple test to make sure we can include, link, and run galario from pure C++.
 */
#include "galario.h"

#include <cassert>
#include <complex>
#include <vector>

using namespace galario;

int main()
{
    init();
    constexpr int nx = 128;
    constexpr int ny = nx;

    // vector initializes to 0
    std::vector<dreal> realdata(nx*ny);
    dcomplex* res = copy_input(nx, ny, &realdata[0]);

    // These checks compute garbage but we want to check if they compile and link.
    // Link errors could come from incompatible const modifiers.
    int n = 4;
    dreal* rp = &realdata[0];
    dcomplex* cp = res;
    dreal r = realdata[0];

    dreal dxy = 0.2;
    sweep(nx, rp, dxy/100., dxy/10.5, nx, dxy, 0.5, cp);
    uv_rotate(r, r, r, rp, rp, n, rp, rp ,rp, rp);

    fft2d(nx, ny, res);
    fftshift(n, n, cp);
    fftshift_axis0(n, n, cp);
    // interpolate(n, n, cp, n, rp, rp, r, cp);
    apply_phase_sampled(r, r, n, rp, rp, cp);

    auto ncomplex = nx*(ny/2+1);
    std::vector<dcomplex> fint(res, res + ncomplex);
    auto chi2 = reduce_chi2(300, &realdata[0], &realdata[0], res, &realdata[0]);
    (void)chi2; // avoid unused variable warning

    // copy input before reduce_chi2 to see if it is modified inadvertently
    for (auto i = 0; i < ncomplex; ++i) {
        assert(fint[i] == res[i]);
    }

    galario_free(res);
    cleanup();

    return 0;
}
