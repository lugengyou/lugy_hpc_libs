/*
 * @Author: lugy lugengyou@github.com
 * @Date: 2024-09-23 23:30:00
 * @FilePath: /lugy_hpc_libs/src/c/transpose.cc
 * @LastEditTime: 2024-09-23 23:42:48
 * @Description: transpose.cc
 */
#include "transpose.h"


void lugy::transpose_c(int *a, int *b, int srcN, int srcM) {
    for (int i = 0; i < srcN; i++) {
        for (int j = 0; j < srcM; j++) {
            b[j*srcN + i] = a[i*srcM + j];
        }
    }
}

