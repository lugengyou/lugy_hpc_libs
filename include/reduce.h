/*
 * @Author: lugy lugengyou@github.com
 * @Date: 2024-09-18 15:32:45
 * @FilePath: /lugy_hpc_libs/include/reduce.h
 * @LastEditTime: 2024-09-18 16:52:23
 * @Description: reduce.h
 */
#include <iostream>

namespace lugy {
    void reduce_c(int *a, int *b, int n);

    void reduce_cuda(int *a, int *b, int n, int version=0);

} // namespace lugy

