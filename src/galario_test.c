/*
 * A simple test to make sure we can include, link, and run galario from pure C.
 */
#include "galario.h"

#include <complex.h>

#define nx 128
#define ny 128

int main() {
     galario_init();

     dreal data[nx*ny];
     galario_fftshift(nx, ny, &data[0]);
     galario_cleanup();

     return 0;
}
