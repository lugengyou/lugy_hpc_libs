/*
 * @Author: lugy lugengyou@github.com
 * @Date: 2024-09-23 23:33:10
 * @FilePath: /lugy_hpc_libs/tests/transpose.cc
 * @LastEditTime: 2024-09-24 00:26:17
 * @Description: transpose test sample
 */
#include "transpose.h"
#include <iostream>


int main() {
    std::cout << "transpose test." << std::endl;

    const int n = 256;
    const int m = 512;

    int a[n*m];
    for (int i = 0; i < n*m; ++i) {
        a[i] = i;
    }
    int b_c[n*m] = {0};
    int b_cuda[n*m] = {0};

    std::cout << "a: " << std::endl;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << a[i * m + j] << " ";
        }
        std::cout << std::endl;
    }

    // call the transpose function of c
    lugy::transpose_c(a, b_c, n, m);

    std::cout << "b_c: " << std::endl;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << b_c[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    // call the transpose function of cuda
    lugy::transpose_cuda(a, b_cuda, n, m, 0);
    std::cout << "b_cuda: " << std::endl;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << b_cuda[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

