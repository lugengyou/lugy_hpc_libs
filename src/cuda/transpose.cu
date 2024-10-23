/*
 * @Author: lugy lugengyou@github.com
 * @Date: 2024-09-23 23:31:04
 * @FilePath: /lugy_hpc_libs/src/cuda/transpose.cu
 * @LastEditTime: 2024-10-23 15:36:42
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

static __global__ void transpose_kernel_v1(int *a, int *b, int srcN, int srcM) {
    __shared__ int sdata[32][32];

    size_t y = threadIdx.y + blockIdx.y * blockDim.y; // 行
    size_t x = threadIdx.x + blockIdx.x * blockDim.x; // 列

    size_t src_addr = y * srcM + x;
    
    size_t share_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // 计算共享内存转置对应的行列
    size_t row = share_idx / blockDim.y;
    size_t col = share_idx % blockDim.y;

    // 转置后的目标矩阵行列
    size_t trans_y = blockIdx.x * blockDim.x + row;
    size_t trans_x = blockIdx.y * blockDim.y + col; // trans_x 是连续增加的

    size_t dst_addr = trans_y * srcN + trans_x;

    if (y < srcN && x < srcM) {
        // 加载数据到共享内存
        sdata[threadIdx.y][threadIdx.x] = a[src_addr];
        __syncthreads();

        // 共享内存按列连续读取，转置按行连续存储     
        b[dst_addr] = sdata[col][row]; 
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
    else if (version == 1) {
        printf("cuda in transpose_kernel_v1\n");
        dim3 block(32, 32);
        dim3 grid((srcM + block.x - 1) / block.x, (srcN + block.y - 1) / block.y);
        transpose_kernel_v1<<<grid, block>>>(d_a, d_b, srcN, srcM);
    }
    else {
        printf("version not found\n");
          
    }

    cudaMemcpy(b, d_b, srcN * srcM * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
}

