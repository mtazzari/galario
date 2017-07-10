/*
 * A simple test to make sure we can include, link, and run galario from pure C.
 */
#include "galario.h"

#include <complex.h>

#define nx 128

int main() {
     galario_acc_init();

     dcomplex data[nx*nx];
     galario_fftshift(nx, &data[0]);
     galario_acc_cleanup();

     return 0;
}
