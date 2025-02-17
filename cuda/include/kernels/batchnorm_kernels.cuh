#pragma once
#include <cuda_runtime.h>

// Batch normalization forward kernel
__global__ void batch_norm_forward_kernel(
    float* input,           // [batch, channels, height, width]
    const float* gamma,     // [channels]
    const float* beta,      // [channels]
    float* running_mean,    // [channels]
    float* running_var,     // [channels]
    float* batch_mean,      // [channels]
    float* batch_var,       // [channels]
    int batch_size,
    int channels,
    int height,
    int width,
    float epsilon,
    float momentum,
    bool is_training
);

// Batch normalization backward kernel
__global__ void batch_norm_backward_kernel(
    const float* grad_output,     // [batch, channels, height, width]
    const float* input,           // [batch, channels, height, width]
    const float* gamma,           // [channels]
    const float* batch_mean,      // [channels]
    const float* batch_var,       // [channels]
    float* grad_input,            // [batch, channels, height, width]
    float* grad_gamma,            // [channels]
    float* grad_beta,             // [channels]
    int batch_size,
    int channels,
    int height,
    int width,
    float epsilon
);