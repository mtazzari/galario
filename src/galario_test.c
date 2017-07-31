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
