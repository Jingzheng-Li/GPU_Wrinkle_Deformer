#pragma once

#include <iostream>
#include <vector>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

// CUDA Initialization
inline void init_CUDA() {
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        exit(EXIT_FAILURE);
    }
}

inline unsigned long long LogIte(unsigned long long value) {
    if (value == 0) {
        return 0;
    }
    return 1 + LogIte(value >> 1);
}

inline unsigned long long Log2(unsigned long long value) {
    value -= 1;
    if (value == 0) {
        return 1;
    }
    return LogIte(value);
}


#define CHECK_ERROR_CUDA(cond, msg, error) \
    if (!(cond)) { \
        printf("\033[1;31mCUDA Error: %s\033[0m\n", msg); \
        error = true; \
        return; \
    }

#define CHECK_ERROR(cond, msg) \
    if (!(cond)) { \
        printf("\033[1;31mCUDA Error: %s\033[0m\n", msg); \
        return; \
    }

#define CUDA_SAFE_CALL(err) cuda_safe_call_(err, __FILE__, __LINE__)
inline void cuda_safe_call_(cudaError_t err, const char* file_name, const int num_line) {
    if (cudaSuccess != err) {
        std::cerr << "\033[1;31m" << file_name << "[" << num_line << "]: "
                  << "CUDA Runtime API error[" << static_cast<int>(err) << "]: "
                  << cudaGetErrorString(err) << "\033[0m" << std::endl;

        exit(EXIT_FAILURE);
    }
}

#define CUDA_KERNEL_CHECK() cuda_kernel_check_(__FILE__, __LINE__)
inline void cuda_kernel_check_(const char* file_name, const int num_line) {
    cudaError_t err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        std::cerr << "\033[1;31m" << file_name << "[" << num_line << "]: "
                  << "CUDA Kernel execution error[" << static_cast<int>(err) << "]: "
                  << cudaGetErrorString(err) << "\033[0m" << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_INDEX_BOUNDS(idx, limit) check_index_bounds_(idx, limit, __FILE__, __LINE__)
__device__ inline void check_index_bounds_(uint32_t idx, uint32_t limit, const char* file_name, int num_line) {
    if (idx >= limit) {
        printf("%s[%d]: Error: index %d out of bounds (>= %d)\n", file_name, num_line, idx, limit);
        asm("trap;");
    }
}


template <typename T>
inline void CUDAMallocSafe(T*& cudaData, int size) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&cudaData, size * sizeof(T)));
    CUDA_SAFE_CALL(cudaMemset(cudaData, 0, size * sizeof(T)));
}

template<typename T>
inline void CUDAFreeSafe(T*& cudaData) {
    if (cudaData) {
        cudaPointerAttributes attributes;
        cudaError_t status = cudaPointerGetAttributes(&attributes, cudaData);
        if (status == cudaSuccess && attributes.type == cudaMemoryTypeDevice) {
            CUDA_SAFE_CALL(cudaFree(cudaData));
            cudaData = nullptr;
        } else {
            std::cerr << "Warning: Attempted to free an invalid or already freed device pointer." << std::endl;
        }
    }
}


// 处理单个对象从主机到设备的内存拷贝
template <typename CudaType>
inline void CUDAMemcpyHToDSafe(CudaType* dstCudaData, const CudaType& srcData) {
    CUDA_SAFE_CALL(cudaMemcpy(dstCudaData, &srcData, sizeof(CudaType), cudaMemcpyHostToDevice));
}

// 处理 std::vector 从主机到设备的内存拷贝
template <typename CudaType>
inline void CUDAMemcpyHToDSafe(CudaType* dstCudaData, const std::vector<CudaType>& srcVectorData) {
    size_t dataSize = srcVectorData.size();
    CUDA_SAFE_CALL(cudaMemcpy(dstCudaData, srcVectorData.data(), dataSize * sizeof(CudaType), cudaMemcpyHostToDevice));
}

// 处理单个对象从设备到主机的内存拷贝
template <typename CudaType>
inline void CUDAMemcpyDToHSafe(CudaType& dstData, const CudaType* srcCudaData) {
    CUDA_SAFE_CALL(cudaMemcpy(&dstData, srcCudaData, sizeof(CudaType), cudaMemcpyDeviceToHost));
}

// 处理 std::vector 从设备到主机的内存拷贝
template <typename CudaType>
inline void CUDAMemcpyDToHSafe(std::vector<CudaType>& dstVectorData, const CudaType* srcCudaData) {
    size_t dataSize = dstVectorData.size();
    CUDA_SAFE_CALL(cudaMemcpy(dstVectorData.data(), srcCudaData, dataSize * sizeof(CudaType), cudaMemcpyDeviceToHost));
}

template <typename CudaType>
inline void CUDAMemcpyDToDSafe(CudaType* dstCudaData, const CudaType* srcCudaData, size_t dataSize) {
    CUDA_SAFE_CALL(cudaMemcpy(dstCudaData, srcCudaData, dataSize * sizeof(CudaType), cudaMemcpyDeviceToDevice));
}


////////////////////////////////////////
// int idof = blockIdx.x * blockDim.x;
// int idx = threadIdx.x + idof;
// if (idx >= numbers) return;
// blockIdx.x represent current block id
// blockDim.x represent threads of perblock default as 256
// threadIdx.x represent current thread of the block 0-255
// warp as 32, each block has 8 warp
// parallel in all threads of all block
/*

+-------------------------+   +-------------------------+
|       Block 0           |   |       Block 1           |
|                         |   |                         |
|  +-------------------+  |   |  +-------------------+  |
|  | Warp 0 (Thread 0-31)|  |   |  | Warp 0 (Thread 0-31)|  |
|  +-------------------+  |   |  +-------------------+  |
|  +-------------------+  |   |  +-------------------+  |
|  | Warp 1 (Thread 32-63)|  |   |  | Warp 1 (Thread 32-63)|  |
|  +-------------------+  |   |  +-------------------+  |
|  +-------------------+  |   |  +-------------------+  |
|  | Warp 2 (Thread 64-95)|  |   |  | Warp 2 (Thread 64-95)|  |
|  +-------------------+  |   |  +-------------------+  |
|  +-------------------+  |   |  +-------------------+  |
|  | Warp 3 (Thread 96-127)|  |   |  | Warp 3 (Thread 96-127)|  |
|  +-------------------+  |   |  +-------------------+  |
|  +-------------------+  |   |  +-------------------+  |
|  | Warp 4 (Thread 128-159)|  |   |  | Warp 4 (Thread 128-159)|  |
|  +-------------------+  |   |  +-------------------+  |
|  +-------------------+  |   |  +-------------------+  |
|  | Warp 5 (Thread 160-191)|  |   |  | Warp 5 (Thread 160-191)|  |
|  +-------------------+  |   |  +-------------------+  |
|  +-------------------+  |   |  +-------------------+  |
|  | Warp 6 (Thread 192-223)|  |   |  | Warp 6 (Thread 192-223)|  |
|  +-------------------+  |   |  +-------------------+  |
|  +-------------------+  |   |  +-------------------+  |
|  | Warp 7 (Thread 224-255)|  |   |  | Warp 7 (Thread 224-255)|  |
|  +-------------------+  |   |  +-------------------+  |
+-------------------------+   +-------------------------+


*/
///////////////////////////////////////
