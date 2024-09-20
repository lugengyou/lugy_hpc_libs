/*
 * @Author: lugy lugengyou@github.com
 * @Date: 2024-09-19 23:13:53
 * @FilePath: /lugy_hpc_libs/src/c/gemm.cc
 * @LastEditTime: 2024-09-20 09:42:49
 * @Description: gemm.cc
 */
#include "gemm.h"
#include <stdio.h>

/**
 * @description: 朴素矩阵乘
 * @return {*}
 */
static void gemm_c_v0(int *a, int *b, int *c, int N, int M, int K) {
    for (int i = 0; i < N; ++i) {        
        for (int j = 0; j < M; ++j) {
            int sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * M + j];
            }
            c[i * M + j] = sum;
        }
    }
}


void lugy::gemm_c(int *a, int *b, int *c, int N, int M, int K, int version) {
    switch (version) {
        case 0:
            printf("in gemm_c_v0\n");
            gemm_c_v0(a, b, c, N, M, K);
            break;
        default:
            printf("no version found\n");
            break;
    }
    
}

