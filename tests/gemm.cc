/*
 * @Author: lugy lugengyou@github.com
 * @Date: 2024-09-20 11:22:12
 * @FilePath: /lugy_hpc_libs/tests/gemm.cc
 * @LastEditTime: 2024-09-20 11:35:26
 * @Description: gemm test sample
 */
#include "gemm.h"
#include <stdio.h>


int main() {
    printf("gemm test.\n");

    const int N = 512;
    const int M = 512;
    const int K = 256;

    int a[N * K];
    for (int i = 0; i < N * K; ++i) {
        a[i] = 1;
    }

    int b[K * M];
    for (int i = 0; i < K * M; ++i) {
        b[i] = 1;
    }

    int c[N * M];
    for (int i = 0; i < N * M; ++i) {
        c[i] = 0;
    }

    int d_c[N * M];
    for (int i = 0; i < N * M; ++i) {
        d_c[i] = 0;
    }
    
    // cpu gemm
    lugy::gemm_c(a, b, c, N, M, K, 0);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            printf("%d ", c[i * M + j]);
        }
        printf("\n");
    }
    
    // gpu gemm
    lugy::gemm_cuda(a, b, d_c, N, M, K, 1);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            printf("%d ", d_c[i * M + j]);
        }
        printf("\n");
    }

    return 0;
}





