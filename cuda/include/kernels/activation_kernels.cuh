#pragma once
#include <cuda_runtime.h>

// ReLU forward kernel
__global__ void relu_kernel(
    float* data,
    int size
);

// ReLU backward kernel
__global__ void relu_backward_kernel(
    const float* grad_output,
    const float* forward_output,
    float* grad_input,
    int size
);