/*
 * @Author: lugy lugengyou@github.com
 * @Date: 2024-09-23 23:27:29
 * @FilePath: /lugy_hpc_libs/include/transpose.h
 * @LastEditTime: 2024-09-23 23:29:47
 * @Description: transpose.h
 */
#ifndef _TRANSPOSE_H_
#define _TRANSPOSE_H_


namespace lugy {
    void transpose_c(int *a, int *b, int srcN, int srcM);

    void transpose_cuda(int *a, int *b, int srcN, int srcM, int version=0);

} // namespace lugy

#endif // _TRANSPOSE_H_

