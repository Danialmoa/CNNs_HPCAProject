#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "adam_optimizer.cuh"

class ConvBlock {
private:
    int in_channels, out_channels, kernel_size, stride, padding;
    int pool_size, pool_stride;
    
    // Device pointers
    float *d_weights, *d_biases;
    float *d_cache, *d_conv_output_cache, *d_relu_output_cache;
    int *d_pool_indices;
    
    float learning_rate;
    
    // Dimensions
    int input_height, input_width;
    int conv_output_height, conv_output_width;
    int pool_output_height, pool_output_width;

    //Adam Optimizer
    AdamOptimizer weights_optimizer;
    AdamOptimizer bias_optimizer;

    // Helper functions
    void allocate_memory(int batch_size);
    void free_memory();

public:
    ConvBlock(int in_channels, int out_channels, int kernel_size, 
              int stride, int padding, int pool_size, int pool_stride, 
              float learning_rate);
    ~ConvBlock();

    // Forward and backward functions
    void forward(const float* d_input, float* d_output, int batch_size, 
                int height, int width);
    void backward(const float* d_grad_output, float* d_grad_input, int batch_size);
};