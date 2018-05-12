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
 * A simple test to make sure we can include, link, and run galario from pure C.
 */
#include "galario.h"

#include <complex.h>

#define nx 128
#define ny 128

int main() {
     galario_init();
     dreal realdata[nx*ny];
     dcomplex* res = galario_copy_input(nx, ny, realdata);
     galario_fft2d(nx, ny, res);
     galario_free(res);
     galario_cleanup();

     return 0;
}
