#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "adam_optimizer.cuh"

class ConvBlock {
private:
    // Layer parameters
    int in_channels, out_channels;
    int kernel_size, stride, padding;
    int pool_size, pool_stride;
    int current_batch_size;
    float learning_rate;
    
    // Dimensions
    int input_height, input_width;
    int conv_output_height, conv_output_width;
    int pool_output_height, pool_output_width;
    
    // Device pointers
    float* d_weights;         // [out_channels, in_channels, kernel, kernel]
    float* d_biases;          // [out_channels]
    float* d_cache;           // Input cache for backward pass [batch, in_channels, height, width]
    float* d_conv_output_cache; // Convolution output cache [batch, out_channels, conv_h, conv_w]
    int* d_pool_indices;      // Pooling indices cache [batch, out_channels, pool_h, pool_w]
    
    // CUDA streams for parallel execution
    cudaStream_t stream1, stream2, stream3;
    bool streams_initialized;
    
    // Optimizers
    AdamOptimizer weights_optimizer;
    AdamOptimizer bias_optimizer;
    
    // Helper functions
    void allocate_memory(int batch_size);
    void free_memory();
    void clear_cache() {
        cudaDeviceSynchronize();
    }
    void init_streams() {
        if (!streams_initialized) {
            CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
            CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));
            CHECK_CUDA_ERROR(cudaStreamCreate(&stream3));
            streams_initialized = true;
        }
    }

public:
    // Constructor and destructor
    ConvBlock(int in_channels, int out_channels, int kernel_size, 
              int stride, int padding, int pool_size, int pool_stride, 
              float learning_rate);
    ~ConvBlock();

    // Forward and backward functions
    void forward(const float* d_input, float* d_output, int batch_size, 
                int height, int width);
    void backward(const float* d_grad_output, float* d_grad_input, int batch_size);

    // Getter for conv output cache (for testing)
    const float* get_conv_output_cache() const { return d_conv_output_cache; }
};