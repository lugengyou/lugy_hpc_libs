/*
 * @Author: lugy lugengyou@github.com
 * @Date: 2024-09-18 15:32:31
 * @FilePath: /lugy_hpc_libs/src/cuda/reduce.cu
 * @LastEditTime: 2024-09-25 22:38:45
 * @Description: reduce.cu
 */
#include "reduce.h"
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>


/**
 * @description: 单 block 共享内存规约
 * @return {*}
 */
__global__ static void reduce_kernel_v0(int *d_a, int *d_b, int n) {
    extern __shared__ int sdata[];

    const uint32_t tid = threadIdx.x;
    
    // 初始化共享内存
    sdata[tid] = 0; 
    
    // 单个线程处理多个 block 数据规约
    for(uint32_t i = tid; i < n; i += blockDim.x) {
        sdata[tid] += d_a[i];
    }
    __syncthreads();

    // block 内部归约
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }

    // 返回结果
    if (tid == 0) {
        *d_b = sdata[0];
    }
}


/**
 * @description: 单 block 共享内存规约, block 内剩余一个 warp 使用 warp shuffle 规约
 * @return {*}
 */
__global__ static void reduce_kernel_v1(int *d_a, int *d_b, int n) {
    extern __shared__ int sdata[];

    const uint32_t tid = threadIdx.x;
    
    // 初始化共享内存
    sdata[tid] = 0; 
    
    // 单个线程处理多个 block 数据规约
    for(uint32_t i = tid; i < n; i += blockDim.x) {
        sdata[tid] += d_a[i];
    }
    __syncthreads();

    // block 内部归约
    for (int i = blockDim.x / 2; i >= 32; i >>= 1) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }

    // 剩余 32 个线程使用 warp shuffle 规约
    int data = sdata[tid];
    for (int i = 16; i >= 1; i >>= 1) {
        data += __shfl_down_sync(0xFFFFFFFF, data, i);
    }

    // 返回结果
    if (tid == 0) {
        *d_b = data;
    }
}


void lugy::reduce_cuda(int *a, int *b, int n, int version) {
    int *d_a, *d_b;
    
    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_b, sizeof(int));
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int), cudaMemcpyHostToDevice);

    if (version == 0){        
        printf("cuda in reduce_kernel_v0.\n");
        dim3 block(256);
        reduce_kernel_v0<<<1, block, block.x * sizeof(int)>>>(d_a, d_b, n);
    }
    else if (version == 1){
        printf("cuda in reduce_kernel_v1.\n");
        dim3 block(256);
        reduce_kernel_v1<<<1, block, block.x * sizeof(int)>>>(d_a, d_b, n);
    }
    else {
        printf("version not found\n");
    }  

    cudaMemcpy(b, d_b, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
}

