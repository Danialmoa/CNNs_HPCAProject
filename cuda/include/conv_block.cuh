#pragma once
#include <cuda_runtime.h>
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
    int input_height{0}, input_width{0};
    int conv_output_height{0}, conv_output_width{0};
    int pool_output_height{0}, pool_output_width{0};
    
    // Device pointers
    float* d_weights;
    float* d_biases;
    float* d_cache;
    float* d_conv_output_cache;
    int* d_pool_indices;
    
    // Batch normalization parameters
    float* d_gamma;
    float* d_beta;
    float* d_running_mean;
    float* d_running_var;
    float* d_batch_mean;
    float* d_batch_var;
    float bn_epsilon;
    float bn_momentum;
    bool is_training;
    
    // CUDA streams
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
    void init_streams();

public:
    ConvBlock(int in_channels, int out_channels, int kernel_size, 
              int stride, int padding, int pool_size, int pool_stride, 
              float learning_rate);
    ~ConvBlock();

    void forward(const float* d_input, float* d_output, 
                int batch_size, int height, int width);
    void backward(const float* d_grad_output, float* d_grad_input, 
                 int batch_size, int height, int width);

    // Getters for testing
    const float* get_weights() const { return d_weights; }
    const float* get_biases() const { return d_biases; }
    const float* get_gamma() const { return d_gamma; }
    const float* get_beta() const { return d_beta; }
};