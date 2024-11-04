#!/bin/bash
###
 # @Author: lugy lugengyou@github.com
 # @Date: 2024-09-18 15:16:04
 # @FilePath: /lugy_hpc_libs/lugy_hpc_libs_build.sh
 # @LastEditTime: 2024-11-04 16:34:18
 # @Description: lugy_hpc_libs 库编译脚本
### 

current_dir=$(cd $(dirname $0); pwd)
echo "current_dir: $current_dir"

# 创建 build 目录
if [ ! -d "${current_dir}/build" ];then
    mkdir -p "${current_dir}/build"
fi

# 编译 lugy_hpc_libs 库
cd ${current_dir}/build
cmake ..
make

# ./reduce
./gemm
# ./transpose


