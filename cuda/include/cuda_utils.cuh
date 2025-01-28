#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA error checking macro
#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
inline void check_cuda_error(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// Kernel launch error checking macro
#define CHECK_LAST_CUDA_ERROR() check_last_cuda_error(__FILE__, __LINE__)
inline void check_last_cuda_error(const char* file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n",
                file, line, static_cast<unsigned int>(error), cudaGetErrorString(error));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}