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
    
    // Input dimensions
    int input_height{0}, input_width{0};  // Initialize to 0
    int conv_output_height{0}, conv_output_width{0};
    int pool_output_height{0}, pool_output_width{0};
    
    // Device pointers
    float* d_weights;         // [out_channels, in_channels, kernel, kernel]
    float* d_biases;          // [out_channels]
    float* d_cache;           // Input cache for backward pass [batch, in_channels, height, width]
    float* d_conv_output_cache; // Convolution output cache [batch, out_channels, conv_h, conv_w]
    int* d_pool_indices;      // Pooling indices cache [batch, out_channels, pool_h, pool_w]
    
    // Batch normalization parameters
    float* d_gamma;           // [out_channels]
    float* d_beta;            // [out_channels]
    float* d_running_mean;    // [out_channels]
    float* d_running_var;     // [out_channels]
    float* d_batch_mean;      // [out_channels]
    float* d_batch_var;       // [out_channels]
    float bn_epsilon;
    float bn_momentum;
    bool is_training;

    // Gradient buffers
    float* d_grad_weights;    // [out_channels, in_channels, kernel, kernel]
    float* d_grad_biases;     // [out_channels]
    
    // CUDA streams for parallel execution
    cudaStream_t stream1, stream2, stream3;
    bool streams_initialized;
    
    // Optimizers
    AdamOptimizer weights_optimizer;
    AdamOptimizer bias_optimizer;
    AdamOptimizer gamma_optimizer;
    AdamOptimizer beta_optimizer;
    
    // Helper functions
    void allocate_memory(int batch_size);
    void free_memory();
    void init_weights_and_optimizers(); 
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
    void backward(const float* d_grad_output, float* d_grad_input, int batch_size, int height, int width);

    // Getter for conv output cache (for testing)
    const float* get_conv_output_cache() const { return d_conv_output_cache; }

    // Getters for weights and biases (for testing)
    const float* get_weights() const { return d_weights; }
    const float* get_biases() const { return d_biases; }
    const float* get_gamma() const { return d_gamma; }
    const float* get_beta() const { return d_beta; }
    const float* get_running_mean() const { return d_running_mean; }
    const float* get_running_var() const { return d_running_var; }
    const float* get_grad_weights() const { return d_grad_weights; }
};