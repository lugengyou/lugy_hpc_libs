/*
 * @Author: lugy lugengyou@github.com
 * @Date: 2024-09-18 15:32:22
 * @FilePath: /lugy_hpc_libs/src/c/reduce.cc
 * @LastEditTime: 2024-09-18 16:45:39
 * @Description: reduce.cc
 */
#include "reduce.h"


void lugy::reduce_c(int *a, int *b, int n) {
    for (int i = 0; i < n; i++) {
        b[0] += a[i];
    }
}
