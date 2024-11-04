/*
 * @Author: lugy lugengyou@github.com
 * @Date: 2024-09-20 09:43:39
 * @FilePath: /lugy_hpc_libs/src/cuda/gemm.cu
 * @LastEditTime: 2024-11-04 16:34:01
 * @Description: gemm.cu
 */
#include "gemm.h"
#include <stdio.h>


/**
 * @description: 一个线程处理目标矩阵的一个元素
 * @return {*}
 */
__global__ static void gemm_cuda_v0(int *d_a, int *d_b, int *d_c, int N, int M, int K) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N * M) {
        int r = idx / M;
        int c = idx % M;
        int sum = 0;
        for (int i = 0; i < K; ++i) {
            sum += d_a[r * K + i] * d_b[i * M + c];
        }
        d_c[idx] = sum;
    }
}
    
/**
 * @description: 棋盘格矩阵乘法，对每个棋盘格块先加载到共享内存，然后计算
 * @return {*}
 */
__global__ static void gemm_cuda_v1(int *d_a, int *d_b, int *d_c, int N, int M, int K) {
    __shared__ int sdataA[16][16];
    __shared__ int sdataB[16][16];

    const int tidc = threadIdx.x;
    const int tidr = threadIdx.y;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int sum = 0;
    // 加载数据到共享内存
    for (int i = 0; i < K; i += 16) {
        sdataA[tidr][tidc] = d_a[idy * K + i + tidc];
        sdataB[tidr][tidc] = d_b[(i + tidr) * M + idx];
        __syncthreads();

        // 计算        
        for (int j = 0; j < 16; ++j) {
            sum += sdataA[tidr][j] * sdataB[j][tidc];
        }
        __syncthreads();
    }

    if (idy < N && idx < M) {
        d_c[idy * M + idx] = sum;
    }   
}


void lugy::gemm_cuda(int *a, int *b, int *c, int N, int M, int K, int version) {
    
    int *d_a, *d_b, *d_c;
    
    cudaMalloc((void**)&d_a, N * K * sizeof(int));
    cudaMalloc((void**)&d_b, K * M * sizeof(int));
    cudaMalloc((void**)&d_c, N * M * sizeof(int));

    cudaMemcpy(d_a, a, N * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * M * sizeof(int), cudaMemcpyHostToDevice);
    
    if (version == 0) {
        printf("in gemm_cuda version 0\n");
        dim3 block(256);
        dim3 grid((N * M + block.x - 1) / block.x);
        gemm_cuda_v0<<<grid, block>>>(d_a, d_b, d_c, N, M, K);
    }
    else if (version == 1) {
        printf("in gemm_cuda version 1\n");
        dim3 block(16, 16);
        dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        gemm_cuda_v1<<<grid, block>>>(d_a, d_b, d_c, N, M, K);
    }
    else {
        printf("version not found\n");
    }

    cudaMemcpy(c, d_c, N * M * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

