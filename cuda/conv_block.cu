#include "include/conv_block.cuh"
#include "include/kernels/conv_kernels.cuh"
#include "include/kernels/activation_kernels.cuh"
#include "include/kernels/pooling_kernels.cuh"
#include "include/kernels/batchnorm_kernels.cuh"
#include <random>
#include <cmath>
#include <iostream>

__global__ void scale_gradients(
    float* weight_grad,
    float* bias_grad,
    float scale,
    int total_weights,
    int num_biases
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_weights) {
        weight_grad[idx] *= scale;
    }
    if (idx < num_biases) {
        bias_grad[idx] *= scale;
    }
}

ConvBlock::ConvBlock(int in_ch, int out_ch, int k_size, 
                     int s, int p, int pool_s, int pool_str, 
                     float lr) 
    : in_channels(in_ch), out_channels(out_ch), kernel_size(k_size),
      stride(s), padding(p), pool_size(pool_s), pool_stride(pool_str),
      learning_rate(lr), current_batch_size(0), streams_initialized(false),
      bn_epsilon(1e-5), bn_momentum(0.1), is_training(true),
      d_cache(nullptr), d_conv_output_cache(nullptr), d_pool_indices(nullptr),
      d_weights(nullptr), d_biases(nullptr), d_gamma(nullptr), d_beta(nullptr),
      d_running_mean(nullptr), d_running_var(nullptr),
      d_batch_mean(nullptr), d_batch_var(nullptr),
      d_pool_grad(nullptr), d_relu_grad(nullptr), d_bn_grad(nullptr),
      d_weight_grad(nullptr), d_bias_grad(nullptr) {
    
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

    if (batch_size != current_batch_size || height != input_height || width != input_width) {
        std::cerr << "Error: Dimensions mismatch in backward pass" << std::endl;
        return;
    }
    //Debug
    std::cout << "Backward pass" << std::endl;
    std::cout << "batch_size: " << batch_size << " height: " << height << " width: " << width << std::endl;
    std::cout << "current_batch_size: " << current_batch_size << " input_height: " << input_height << " input_width: " << input_width << std::endl;
    std::cout << "out_channels: " << out_channels << " conv_output_height: " << conv_output_height << " conv_output_width: " << conv_output_width << std::endl;
    std::cout << "pool_output_height: " << pool_output_height << " pool_output_width: " << pool_output_width << std::endl;
    std::cout << "in_channels: " << in_channels << " kernel_size: " << kernel_size << " stride: " << stride << " padding: " << padding << std::endl;
    std::cout << "pool_size: " << pool_size << " pool_stride: " << pool_stride << std::endl;

    if (d_pool_grad == nullptr || d_relu_grad == nullptr || d_bn_grad == nullptr) {
        allocate_memory(batch_size);
    }

    // Initialize gradients to zero
    size_t conv_output_size = batch_size * out_channels * conv_output_height * conv_output_width;
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_pool_grad, 0, conv_output_size * sizeof(float), stream1));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_relu_grad, 0, conv_output_size * sizeof(float), stream2));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_bn_grad, 0, conv_output_size * sizeof(float), stream2));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_grad_input, 0, 
        batch_size * in_channels * height * width * sizeof(float), stream3));

    const float grad_scale = 1.0f / (batch_size * out_channels);
    
    // 1. Max Pooling Backward
    dim3 pool_block(16, 16);
    dim3 pool_grid(batch_size, out_channels, 
                   pool_output_height * pool_output_width);

    max_pool_backward_kernel<<<pool_grid, pool_block, 0, stream1>>>(
        d_grad_output,
        d_pool_grad,  // Gradients for conv output
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
    cudaStreamSynchronize(stream1);

    std::cout << "Pooling backward done" << std::endl;

    // 2. ReLU Backward
    relu_backward_kernel<<<conv_output_size, 256, 0, stream2>>>(
        d_pool_grad,  // Grad from pooling
        d_conv_output_cache,  // Forward output (cached)
        d_relu_grad,
        conv_output_size
    );
    cudaStreamSynchronize(stream2);
    std::cout << "ReLU backward done" << std::endl;
    // 3. Batch Normalization Backward
    const int threads_per_block = 256;
    batch_norm_backward_kernel<<<out_channels, threads_per_block, 0, stream2>>>(
        d_relu_grad,
        d_conv_output_cache,  // Original input to BN
        d_gamma,
        d_batch_mean,
        d_batch_var,
        d_bn_grad,
        d_gamma_grad,
        d_beta_grad,
        batch_size,
        out_channels,
        conv_output_height,
        conv_output_width,
        bn_epsilon
    );
    cudaStreamSynchronize(stream2);
    std::cout << "Batch normalization backward done" << std::endl;
    // 4. Convolution Backward
    dim3 conv_block(16, 16);
    dim3 conv_grid(
        (conv_output_width + conv_block.x - 1) / conv_block.x,
        (conv_output_height + conv_block.y - 1) / conv_block.y,
        batch_size * out_channels
    );
    conv_backward_kernel<<<conv_grid, conv_block, 0, stream3>>>(
        d_bn_grad,  // Grad from BN
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
    cudaStreamSynchronize(stream3);
    std::cout << "Convolution backward done" << std::endl;
    const int block_size = 256;
    const int total_params = out_channels * in_channels * kernel_size * kernel_size;
    const int num_blocks_scale = (total_params + block_size - 1) / block_size;
    
    scale_gradients<<<num_blocks_scale, block_size, 0, stream3>>>(
        d_weight_grad,
        d_bias_grad,
        grad_scale,
        total_params,
        out_channels
    );
    cudaStreamSynchronize(stream3);
    std::cout << "Scale gradients done" << std::endl;
    // Update parameters using Adam optimizer
    weights_optimizer.update(d_weights, d_weight_grad);
    bias_optimizer.update(d_biases, d_bias_grad);
    gamma_optimizer.update(d_gamma, d_gamma_grad);
    beta_optimizer.update(d_beta, d_beta_grad);
    std::cout << "Parameters updated" << std::endl;
    // Synchronize all streams
    std::cout << "Synchronizing streams" << std::endl;
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream3));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void ConvBlock::init_weights_and_optimizers() {
    // Calculate sizes
    size_t weights_size = out_channels * in_channels * kernel_size * kernel_size;
    size_t bias_size = out_channels;

    // Xavier/Glorot initialization for weights
    float weight_bound = sqrt(6.0f / (in_channels + out_channels));
    std::vector<float> h_weights(weights_size);
    std::vector<float> h_biases(bias_size, 0.0f);
    std::vector<float> h_gamma(out_channels, 1.0f);
    std::vector<float> h_beta(out_channels, 0.0f);
    std::vector<float> h_running_stats(out_channels, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> weight_dist(-weight_bound, weight_bound);

    for (size_t i = 0; i < weights_size; ++i) {
        h_weights[i] = weight_dist(gen);
    }
    std::cout << "Allocating memory" << std::endl;

    CHECK_CUDA_ERROR(cudaMalloc(&d_weights, weights_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_biases, bias_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weight_grad, weights_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bias_grad, bias_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_gamma, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_beta, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_gamma_grad, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_beta_grad, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_running_mean, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_running_var, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_mean, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_var, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, h_weights.data(), weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_biases, h_biases.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_gamma, h_gamma.data(), out_channels * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_beta, h_beta.data(), out_channels * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_running_mean, h_running_stats.data(), out_channels * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_running_var, h_running_stats.data(), out_channels * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize optimizers
    std::cout << "Initializing optimizers" << std::endl;
    weights_optimizer = AdamOptimizer(weights_size, learning_rate);
    std::cout << "Weights optimizer initialized" << std::endl;
    bias_optimizer = AdamOptimizer(bias_size, learning_rate);
    std::cout << "Bias optimizer initialized" << std::endl;
    gamma_optimizer = AdamOptimizer(out_channels, learning_rate);
    std::cout << "Gamma optimizer initialized" << std::endl;
    beta_optimizer = AdamOptimizer(out_channels, learning_rate);
    std::cout << "Beta optimizer initialized" << std::endl;

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream3));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << "Initialization done" << std::endl;
}

void ConvBlock::init_streams() {
    if (!streams_initialized) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream3));
        streams_initialized = true;
    }
}

void ConvBlock::allocate_memory(int batch_size) {
    if (current_batch_size != batch_size || d_conv_output_cache == nullptr) {
        std::cout << "Allocating memory" << std::endl;
        free_memory();
        
        // Calculate sizes
        size_t input_size = batch_size * in_channels * input_height * input_width;
        size_t conv_output_size = batch_size * out_channels * conv_output_height * conv_output_width;
        size_t pool_indices_size = batch_size * out_channels * pool_output_height * pool_output_width;

        // Allocate forward pass memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_cache, input_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_conv_output_cache, conv_output_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_pool_indices, pool_indices_size * sizeof(int)));

        // Allocate backward pass memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_pool_grad, conv_output_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_relu_grad, conv_output_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_bn_grad, conv_output_size * sizeof(float)));
        
        size_t weight_grad_size = out_channels * in_channels * kernel_size * kernel_size;
        CHECK_CUDA_ERROR(cudaMalloc(&d_weight_grad, weight_grad_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_bias_grad, out_channels * sizeof(float)));

        current_batch_size = batch_size;
    }
}

void ConvBlock::free_memory() {
    // Free forward pass memory
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

    // Free backward pass memory
    if (d_pool_grad) {
        cudaFree(d_pool_grad);
        d_pool_grad = nullptr;
    }
    if (d_relu_grad) {
        cudaFree(d_relu_grad);
        d_relu_grad = nullptr;
    }
    if (d_bn_grad) {
        cudaFree(d_bn_grad);
        d_bn_grad = nullptr;
    }
    if (d_weight_grad) {
        cudaFree(d_weight_grad);
        d_weight_grad = nullptr;
    }
    if (d_bias_grad) {
        cudaFree(d_bias_grad);
        d_bias_grad = nullptr;
    }

    current_batch_size = 0;
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