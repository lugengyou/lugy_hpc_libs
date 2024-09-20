/*
 * @Author: lugy lugengyou@github.com
 * @Date: 2024-09-19 23:16:46
 * @FilePath: /lugy_hpc_libs/include/gemm.h
 * @LastEditTime: 2024-09-20 09:36:25
 * @Description: gemm.h
 */
#ifndef _GEMM_H_
#define _GEMM_H_

namespace lugy {
    void gemm_c(int *a, int *b, int *c, int N, int M, int K, int version=0);

    void gemm_cuda(int *a, int *b, int *c, int N, int M, int K, int version=0);

} // namespace lugy

#endif // _GEMM_H_

