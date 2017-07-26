/*
 * A simple test to make sure we can include, link, and run galario from pure C.
 */
#include "galario.h"

#include <complex.h>

#define nx 128

int main() {
     galario_init();

     dreal data[nx*nx];
     galario_fftshift(nx, &data[0]);
     galario_cleanup();

     return 0;
}
