/*
 * @Author: lugy lugengyou@github.com
 * @Date: 2024-09-23 23:31:04
 * @FilePath: /lugy_hpc_libs/src/cuda/transpose.cu
 * @LastEditTime: 2024-09-24 00:27:41
 * @Description: transpose.cu
 */
#include "transpose.h"
#include <stdio.h>
#include <cuda_runtime.h>


static __global__ void transpose_kernel_v0(int *a, int *b, int srcN, int srcM) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < srcN && c < srcM) {
        b[c * srcN + r] = a[r * srcM + c];
    }
}


void lugy::transpose_cuda(int *a, int *b, int srcN, int srcM, int version) {
    
    int *d_a, *d_b;
    
    cudaMalloc((void **)&d_a, srcN * srcM * sizeof(int));
    cudaMalloc((void **)&d_b, srcN * srcM * sizeof(int));
    cudaMemcpy(d_a, a, srcN * srcM * sizeof(int), cudaMemcpyHostToDevice);

    if (version == 0) {
        printf("cuda in transpose_kernel_v0\n");
        dim3 block(32, 32);
        dim3 grid((srcM + block.x - 1) / block.x, (srcN + block.y - 1) / block.y);            
        transpose_kernel_v0<<<grid, block>>>(d_a, d_b, srcN, srcM);
    }
    else {
        printf("version not found\n");
          
    }

    cudaMemcpy(b, d_b, srcN * srcM * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
}

