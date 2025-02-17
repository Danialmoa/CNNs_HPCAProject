#include "include/conv_block.cuh"
#include "include/kernels/conv_kernels.cuh"
#include "include/kernels/activation_kernels.cuh"
#include "include/kernels/pooling_kernels.cuh"
#include "include/kernels/batchnorm_kernels.cuh"
#include <random>
#include <cmath>
#include <iostream>

ConvBlock::ConvBlock(int in_ch, int out_ch, int k_size, 
                     int s, int p, int pool_s, int pool_str, 
                     float lr) 
    : in_channels(in_ch), out_channels(out_ch), kernel_size(k_size),
      stride(s), padding(p), pool_size(pool_s), pool_stride(pool_str),
      learning_rate(lr), current_batch_size(0), streams_initialized(false),
      bn_epsilon(1e-5), bn_momentum(0.1), is_training(true) {
    
    init_weights_and_optimizers();
    init_streams();
}

void ConvBlock::forward(const float* d_input, float* d_output, 
                       int batch_size, int height, int width) {
    // Store dimensions
    input_height = height;
    input_width = width;
    current_batch_size = batch_size;
    
    // Calculate output dimensions
    conv_output_height = (height + 2 * padding - kernel_size) / stride + 1;
    conv_output_width = (width + 2 * padding - kernel_size) / stride + 1;
    pool_output_height = (conv_output_height - pool_size) / pool_stride + 1;
    pool_output_width = (conv_output_width - pool_size) / pool_stride + 1;

    // Log layer info
    std::cout << "\nConvBlock memory allocation:" << std::endl;
    std::cout << "Input shape: " << batch_size << "x" << in_channels << "x" << height << "x" << width << std::endl;
    std::cout << "Conv output shape: " << batch_size << "x" << out_channels << "x" 
              << conv_output_height << "x" << conv_output_width << std::endl;
    std::cout << "Pool output shape: " << batch_size << "x" << out_channels << "x" 
              << pool_output_height << "x" << pool_output_width << std::endl;

    // Allocate memory if needed
    allocate_memory(batch_size);

    // Cache input for backward pass
    cudaMemcpyAsync(d_cache, d_input, 
        batch_size * in_channels * height * width * sizeof(float), 
        cudaMemcpyDeviceToDevice, stream1);

    // 1. Convolution
    dim3 conv_block(16, 16);
    dim3 conv_grid(
        (conv_output_width + conv_block.x - 1) / conv_block.x,
        (conv_output_height + conv_block.y - 1) / conv_block.y,
        batch_size * out_channels
    );

    size_t shared_mem_size = 
        ((conv_block.y + kernel_size - 1) * (conv_block.x + kernel_size - 1) + 
         kernel_size * kernel_size) * sizeof(float);

    conv_forward_kernel<<<conv_grid, conv_block, shared_mem_size, stream1>>>(
        d_input, d_weights, d_biases, d_conv_output_cache,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding, conv_output_height, conv_output_width
    );

    // 2. Batch Normalization
    const int threads_per_block = 256;
    size_t bn_shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    batch_norm_forward_kernel<<<out_channels, threads_per_block, bn_shared_mem_size, stream2>>>(
        d_conv_output_cache, d_gamma, d_beta,
        d_running_mean, d_running_var,
        d_batch_mean, d_batch_var,
        batch_size, out_channels,
        conv_output_height, conv_output_width,
        bn_epsilon, bn_momentum, is_training
    );

    // 3. ReLU
    const int total_elements = batch_size * out_channels * conv_output_height * conv_output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    relu_kernel<<<num_blocks, block_size, 0, stream2>>>(
        d_conv_output_cache, total_elements
    );

    // 4. Max Pooling
    dim3 pool_block(16, 16);
    dim3 pool_grid(
        (pool_output_width + pool_block.x - 1) / pool_block.x,
        (pool_output_height + pool_block.y - 1) / pool_block.y,
        batch_size * out_channels
    );

    max_pool_kernel<<<pool_grid, pool_block, 0, stream3>>>(
        d_conv_output_cache, d_output, d_pool_indices,
        batch_size, out_channels,
        conv_output_height, conv_output_width,
        pool_size, pool_stride,
        pool_output_height, pool_output_width
    );

    // Synchronize all streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
}


void ConvBlock::backward(const float* d_grad_output, float* d_grad_input,
                        int batch_size, int height, int width) {
    // 1. Max Pooling Backward
    dim3 pool_block(16, 16);
    dim3 pool_grid(batch_size, out_channels, 
                   pool_output_height * pool_output_width);

    // Zero initialize gradients
    cudaMemsetAsync(d_conv_output_cache, 0, 
        batch_size * out_channels * conv_output_height * conv_output_width * sizeof(float),
        stream1);

    max_pool_backward_kernel<<<pool_grid, pool_block, 0, stream1>>>(
        d_grad_output,
        d_conv_output_cache,  // Gradients for conv output
        d_pool_indices,
        batch_size,
        out_channels,
        conv_output_height,
        conv_output_width,
        pool_size,
        pool_stride,
        pool_output_height,
        pool_output_width
    );

    // 2. ReLU Backward
    const int total_elements = batch_size * out_channels * conv_output_height * conv_output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    float* d_relu_grad = nullptr;
    float* d_gamma_grad = nullptr;
    float* d_beta_grad = nullptr;
    float* d_weight_grad = nullptr;
    float* d_bias_grad = nullptr;

    // Allocate if not already allocated
    if (!d_relu_grad) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_relu_grad, total_elements * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_gamma_grad, out_channels * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_beta_grad, out_channels * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_weight_grad, 
            out_channels * in_channels * kernel_size * kernel_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_bias_grad, out_channels * sizeof(float)));
    }

    relu_backward_kernel<<<num_blocks, block_size, 0, stream2>>>(
        d_conv_output_cache,  // Grad from pooling
        d_conv_output_cache,  // Forward output (cached)
        d_relu_grad,
        total_elements
    );

    // 3. Batch Normalization Backward
    const int threads_per_block = 256;
    batch_norm_backward_kernel<<<out_channels, threads_per_block, 0, stream2>>>(
        d_relu_grad,
        d_conv_output_cache,  // Original input to BN
        d_gamma,
        d_batch_mean,
        d_batch_var,
        d_conv_output_cache,  // Reuse buffer for BN gradients
        d_gamma_grad,
        d_beta_grad,
        batch_size,
        out_channels,
        conv_output_height,
        conv_output_width,
        bn_epsilon
    );

    // 4. Convolution Backward
    dim3 conv_block(16, 16);
    dim3 conv_grid(
        (conv_output_width + conv_block.x - 1) / conv_block.x,
        (conv_output_height + conv_block.y - 1) / conv_block.y,
        batch_size * out_channels
    );

    conv_backward_kernel<<<conv_grid, conv_block, 0, stream3>>>(
        d_conv_output_cache,  // Grad from BN
        d_cache,             // Cached input
        d_weights,
        d_grad_input,
        d_weight_grad,
        d_bias_grad,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        conv_output_height,
        conv_output_width
    );

    // Update parameters using Adam optimizer
    weights_optimizer.update(d_weights, d_weight_grad);
    bias_optimizer.update(d_biases, d_bias_grad);
    gamma_optimizer.update(d_gamma, d_gamma_grad);
    beta_optimizer.update(d_beta, d_beta_grad);

    // Synchronize all streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
}

void ConvBlock::init_weights_and_optimizers() {
    // Calculate sizes
    size_t weights_size = out_channels * in_channels * kernel_size * kernel_size;
    size_t bias_size = out_channels;

    // Xavier/Glorot initialization for weights
    float weight_bound = sqrt(6.0f / (in_channels + out_channels));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> weight_dist(-weight_bound, weight_bound);

    // Allocate host memory and initialize weights
    std::vector<float> h_weights(weights_size);
    std::vector<float> h_biases(bias_size, 0.0f);  // Initialize biases to zero

    for (size_t i = 0; i < weights_size; ++i) {
        h_weights[i] = weight_dist(gen);
    }

    // Allocate and copy to device memory
    cudaMalloc(&d_weights, weights_size * sizeof(float));
    cudaMalloc(&d_biases, bias_size * sizeof(float));
    cudaMemcpy(d_weights, h_weights.data(), weights_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, h_biases.data(), bias_size * sizeof(float), 
               cudaMemcpyHostToDevice);

    // Initialize batch norm parameters
    cudaMalloc(&d_gamma, out_channels * sizeof(float));
    cudaMalloc(&d_beta, out_channels * sizeof(float));
    cudaMalloc(&d_running_mean, out_channels * sizeof(float));
    cudaMalloc(&d_running_var, out_channels * sizeof(float));
    cudaMalloc(&d_batch_mean, out_channels * sizeof(float));
    cudaMalloc(&d_batch_var, out_channels * sizeof(float));

    std::vector<float> h_gamma(out_channels, 1.0f);
    std::vector<float> h_beta(out_channels, 0.0f);
    std::vector<float> h_running_stats(out_channels, 0.0f);

    cudaMemcpy(d_gamma, h_gamma.data(), out_channels * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta.data(), out_channels * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_running_mean, h_running_stats.data(), 
               out_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_running_var, h_running_stats.data(), 
               out_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize optimizers
    weights_optimizer = AdamOptimizer(weights_size, learning_rate);
    bias_optimizer = AdamOptimizer(bias_size, learning_rate);
    gamma_optimizer = AdamOptimizer(out_channels, learning_rate);
    beta_optimizer = AdamOptimizer(out_channels, learning_rate);
}

void ConvBlock::init_streams() {
    if (!streams_initialized) {
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);
        streams_initialized = true;
    }
}

void ConvBlock::allocate_memory(int batch_size) {
    if (current_batch_size == batch_size && d_conv_output_cache != nullptr) {
        return;  // Memory already allocated with correct size
    }

    // Free existing memory if any
    free_memory();

    // Calculate sizes
    size_t input_size = batch_size * in_channels * input_height * input_width;
    size_t conv_output_size = batch_size * out_channels * conv_output_height * conv_output_width;
    size_t pool_indices_size = batch_size * out_channels * pool_output_height * pool_output_width;

    // Calculate memory requirements in MB
    float input_mb = (input_size * sizeof(float)) / (1024.0f * 1024.0f);
    float conv_output_mb = (conv_output_size * sizeof(float)) / (1024.0f * 1024.0f);
    float pool_indices_mb = (pool_indices_size * sizeof(int)) / (1024.0f * 1024.0f);

    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);

    std::cout << "ConvBlock memory requirements:" << std::endl;
    std::cout << "Input cache: " << input_mb << " MB" << std::endl;
    std::cout << "Conv output cache: " << conv_output_mb << " MB" << std::endl;
    std::cout << "Pool indices: " << pool_indices_mb << " MB" << std::endl;
    std::cout << "Available GPU memory: " << (free_memory / 1024.0f / 1024.0f) << " MB" << std::endl;


    // Allocate memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_cache, input_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv_output_cache, conv_output_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pool_indices, pool_indices_size * sizeof(int)));

    current_batch_size = batch_size;
}

void ConvBlock::free_memory() {
    if (d_cache) {
        cudaFree(d_cache);
        d_cache = nullptr;
    }
    if (d_conv_output_cache) {
        cudaFree(d_conv_output_cache);
        d_conv_output_cache = nullptr;
    }
    if (d_pool_indices) {
        cudaFree(d_pool_indices);
        d_pool_indices = nullptr;
    }
}

ConvBlock::~ConvBlock() {
    free_memory();
    
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_running_mean);
    cudaFree(d_running_var);
    cudaFree(d_batch_mean);
    cudaFree(d_batch_var);

    if (streams_initialized) {
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream3);
    }
}