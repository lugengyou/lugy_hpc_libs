cmake_minimum_required(VERSION 3.0)

project(lugy_hpc_libs)

# lugy_hpc_libs
include_directories(${CMAKE_SOURCE_DIR}/include)

# cuda 库相关
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

link_libraries(${CUDA_LIBRARIES})

file(GLOB SOURCES 
${CMAKE_SOURCE_DIR}/src/c/*.cc 
${CMAKE_SOURCE_DIR}/src/cuda/*.cu)

file(GLOB TEST_SOURCES 
${CMAKE_SOURCE_DIR}/tests/*.cc)

# 循环打印每个测试源文件名
foreach(TEST_FILE ${TEST_SOURCES})
    # 获取算子文件名
    get_filename_component(OP_NAME ${TEST_FILE} NAME_WE) 
    message(STATUS "Build executable program: ${OP_NAME}")
    cuda_add_executable(${OP_NAME} ${SOURCES} ${TEST_FILE})
endforeach()

