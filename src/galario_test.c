/*
 * A simple test to make sure we can include, link, and run galario from pure C.
 */
#include "galario.h"

int main() {
     galario_acc_init();

     galario_acc_cleanup();

     return 0;
}
