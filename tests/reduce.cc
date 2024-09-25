/*
 * @Author: lugy lugengyou@github.com
 * @Date: 2024-09-18 16:13:55
 * @FilePath: /lugy_hpc_libs/tests/reduce.cc
 * @LastEditTime: 2024-09-25 22:37:05
 * @Description: reduce test sample
 */
#include "reduce.h"
#include <iostream>


int main() {
    std::cout << "reduce test." << std::endl;

    const int n = 1 << 20;

    int a[n];
    for (int i = 0; i < n; ++i) {
        a[i] = 1;
    }
    int b_c[1] = {0};
    int b_cuda[1] = {0};

    // call the reduce function of c
    lugy::reduce_c(a, b_c, n);
    std::cout << "b_c[0] = " << b_c[0] << std::endl;

    // call the reduce function of cuda
    lugy::reduce_cuda(a, b_cuda, n, 0);
    std::cout << "b_cuda[0] = " << b_cuda[0] << std::endl;

    return 0;
}





