#pragma once
#include <cudnn.h>
#include "adam_optimizer.cuh"

class ConvBlock {
public:
    ConvBlock(int in_channels, int out_channels, int kernel_size, 
              int stride, int padding, int pool_size, int pool_stride, 
              float learning_rate);
    ~ConvBlock();
    
    void forward(const float* d_input, float* d_output, 
                int batch_size, int height, int width);
    void backward(const float* d_grad_output, float* d_grad_input, 
                 int batch_size);

private:
    // Layer parameters
    int in_channels, out_channels;
    int kernel_size, stride, padding;
    int pool_size, pool_stride;
    float learning_rate;
    
    // Dimensions
    int current_batch_size;
    int input_height, input_width;
    int conv_output_height, conv_output_width;
    int pool_output_height, pool_output_width;
    
    // GPU memory
    float *d_weights, *d_biases;
    float *d_cache, *d_conv_output_cache;
    float *d_workspace;
    size_t workspace_size;
    
    // cuDNN handles and descriptors
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnTensorDescriptor_t conv_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnPoolingDescriptor_t pooling_descriptor;
    cudnnActivationDescriptor_t activation_descriptor;
    
    // cuDNN algorithms
    cudnnConvolutionFwdAlgo_t conv_algorithm;
    cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo;
    
    // CUDA streams
    cudaStream_t stream1, stream2;
    bool streams_initialized;
    
    // Optimizers
    AdamOptimizer weights_optimizer;
    AdamOptimizer bias_optimizer;
    
    void allocate_memory(int batch_size);
    void free_memory();
    void init_streams();
};