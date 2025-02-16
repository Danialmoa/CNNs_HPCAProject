#include "include/conv_block.cuh"
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>


// convolution forward kernel
__global__ void conv_forward_kernel(
    const float* input,           // [batch, in_channels, height, width]
    const float* weights,         // [out_channels, in_channels, kernel, kernel]
    const float* biases,          // [out_channels]
    float* output,                // [batch, out_channels, out_height, out_width]
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int out_height,
    int out_width
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    const int w_out = idx % out_width;
    const int h_out = (idx / out_width) % out_height;
    const int out_ch = (idx / (out_width * out_height)) % out_channels;
    const int batch = idx / (out_width * out_height * out_channels);
    
    float sum = biases[out_ch];
    
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = ((batch * in_channels + in_ch) * height + h_in) * width + w_in;
                    int weight_idx = ((out_ch * in_channels + in_ch) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    output[idx] = sum;
}

// ReLU kernel with 1D indexing
__global__ void relu_kernel(
    float* data,
    int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

// Max pooling kernel with 1D indexing
__global__ void max_pool_kernel(
    const float* input,
    float* output,
    int* indices,
    int batch_size,
    int channels,
    int height,
    int width,
    int pool_size,
    int stride,
    int out_height,
    int out_width
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    const int w_out = idx % out_width;
    const int h_out = (idx / out_width) % out_height;
    const int ch = (idx / (out_width * out_height)) % channels;
    const int batch = idx / (out_width * out_height * channels);
    
    float max_val = -INFINITY;
    int max_idx = -1;
    
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int h_in = h_out * stride + ph;
            int w_in = w_out * stride + pw;
            
            if (h_in < height && w_in < width) {
                int in_idx = ((batch * channels + ch) * height + h_in) * width + w_in;
                float val = input[in_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = in_idx;
                }
            }
        }
    }
    
    output[idx] = max_val;
    indices[idx] = max_idx;
}

// Max pooling backward kernel
__global__ void max_pool_backward_kernel(
    const float* grad_output,     // [batch, channels, pool_out_h, pool_out_w]
    float* grad_input,            // [batch, channels, conv_h, conv_w]
    const int* indices,           // [batch, channels, pool_out_h, pool_out_w]
    int batch_size,
    int channels,
    int conv_height,
    int conv_width,
    int pool_size,
    int stride,
    int out_height,
    int out_width
) {
    int batch = blockIdx.x;
    int channel = blockIdx.y;
    int h = blockIdx.z / out_width;
    int w = blockIdx.z % out_width;
    
    if (batch >= batch_size || channel >= channels || 
        h >= out_height || w >= out_width) return;
        
    int out_idx = ((batch * channels + channel) * out_height + h) * out_width + w;
    int in_idx = indices[out_idx];
    
    // Propagate gradient through max pooling
    atomicAdd(&grad_input[in_idx], grad_output[out_idx]);
}

// ReLU backward kernel
__global__ void relu_backward_kernel(
    const float* grad_output,
    const float* forward_output,
    float* grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // ReLU derivative: 1 if input > 0, 0 otherwise
        grad_input[idx] = grad_output[idx] * (forward_output[idx] > 0 ? 1.0f : 0.0f);
    }
}

// Convolution backward kernel
__global__ void conv_backward_kernel(
    const float* grad_output,     // [batch, out_ch, out_h, out_w]
    const float* input,           // [batch, in_ch, in_h, in_w]
    const float* weights,         // [out_ch, in_ch, kernel, kernel]
    float* grad_input,           // [batch, in_ch, in_h, in_w]
    float* grad_weights,         // [out_ch, in_ch, kernel, kernel]
    float* grad_biases,          // [out_ch]
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int out_height,
    int out_width
) {
    int batch = blockIdx.x;
    int out_ch = blockIdx.y;
    int h = blockIdx.z / out_width;
    int w = blockIdx.z % out_width;
    
    if (batch >= batch_size || out_ch >= out_channels || 
        h >= out_height || w >= out_width) return;
    
    int out_idx = ((batch * out_channels + out_ch) * out_height + h) * out_width + w;
    float grad = grad_output[out_idx];
    
    // Accumulate bias gradients
    atomicAdd(&grad_biases[out_ch], grad);
    
    // Compute gradients for weights and input
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h * stride - padding + kh;
                int w_in = w * stride - padding + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    // Input index
                    int in_idx = ((batch * in_channels + in_ch) * height + h_in) * width + w_in;
                    // Weight index
                    int weight_idx = ((out_ch * in_channels + in_ch) * kernel_size + kh) * kernel_size + kw;
                    
                    // Gradient w.r.t. weights
                    atomicAdd(&grad_weights[weight_idx], input[in_idx] * grad);
                    
                    // Gradient w.r.t. input
                    atomicAdd(&grad_input[in_idx], weights[weight_idx] * grad);
                }
            }
        }
    }
}

// Constructor
ConvBlock::ConvBlock(int in_ch, int out_ch, int k_size, 
                     int s, int p, int pool_s, int pool_str, 
                     float lr) 
    : in_channels(in_ch), out_channels(out_ch), kernel_size(k_size),
      stride(s), padding(p), pool_size(pool_s), pool_stride(pool_str),
      learning_rate(lr), current_batch_size(0), streams_initialized(false),
      weights_optimizer(lr), bias_optimizer(lr) {
    
    // Initialize weights and biases
    std::vector<float> h_weights(out_channels * in_channels * kernel_size * kernel_size);
    std::vector<float> h_biases(out_channels, 0.0f);
    
    // Xavier initialization
    float std_dev = sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (float& w : h_weights) {
        w = dist(gen);
    }
    
    // Allocate and copy weights and biases to GPU
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights, h_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_biases, h_biases.size() * sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, h_weights.data(), 
                               h_weights.size() * sizeof(float), 
                               cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_biases, h_biases.data(), 
                               h_biases.size() * sizeof(float), 
                               cudaMemcpyHostToDevice));
}

// Forward pass
void ConvBlock::forward(const float* d_input, float* d_output, 
                       int batch_size, int height, int width) {
    // Set dimensions
    input_height = height;
    input_width = width;
    conv_output_height = (height + 2 * padding - kernel_size) / stride + 1;
    conv_output_width = (width + 2 * padding - kernel_size) / stride + 1;
    pool_output_height = (conv_output_height - pool_size) / pool_stride + 1;
    pool_output_width = (conv_output_width - pool_size) / pool_stride + 1;
    
    if (batch_size != current_batch_size) {
        allocate_memory(batch_size);
    }

    // Cache input for backward pass
    CHECK_CUDA_ERROR(cudaMemcpy(d_cache, d_input, 
        batch_size * in_channels * height * width * sizeof(float), 
        cudaMemcpyDeviceToDevice));

    // 1. Convolution
    const int threads_per_block = 256;
    const int total_elements = batch_size * out_channels * conv_output_height * conv_output_width;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    std::cout << "num_blocks: " << num_blocks << std::endl;
    std::cout << "threads_per_block: " << threads_per_block << std::endl;
    std::cout << "total_elements: " << total_elements << std::endl;
    conv_forward_kernel<<<num_blocks, threads_per_block>>>(
        d_input, d_weights, d_biases, d_conv_output_cache,
        batch_size, in_channels, out_channels,
        height, width, kernel_size, stride, padding,
        conv_output_height, conv_output_width
    );
    CHECK_LAST_CUDA_ERROR();
    
    // 2. ReLU
    relu_kernel<<<num_blocks, threads_per_block>>>(
        d_conv_output_cache, total_elements
    );
    CHECK_LAST_CUDA_ERROR();
    
    // 3. Max Pooling
    const int pool_elements = batch_size * out_channels * pool_output_height * pool_output_width;
    const int pool_blocks = (pool_elements + threads_per_block - 1) / threads_per_block;
    
    max_pool_kernel<<<pool_blocks, threads_per_block>>>(
        d_conv_output_cache, d_output, d_pool_indices,
        batch_size, out_channels,
        conv_output_height, conv_output_width,
        pool_size, pool_stride,
        pool_output_height, pool_output_width
    );
    CHECK_LAST_CUDA_ERROR();
}

// Backward pass implementation
void ConvBlock::backward(const float* d_grad_output, float* d_grad_input, int batch_size) {
    if (!streams_initialized) {
        init_streams();
    }
    
    // Allocate temporary gradients
    float *d_grad_weights, *d_grad_biases;
    size_t weight_size = out_channels * in_channels * kernel_size * kernel_size;
    size_t conv_size = batch_size * out_channels * conv_output_height * conv_output_width;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_weights, weight_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_biases, out_channels * sizeof(float)));
    
    // Zero out gradients
    CHECK_CUDA_ERROR(cudaMemset(d_grad_weights, 0, weight_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_biases, 0, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_input, 0, batch_size * in_channels * input_height * input_width * sizeof(float)));
    
    // Temporary buffer for gradients
    float* d_conv_grad;
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv_grad, conv_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_conv_grad, 0, conv_size * sizeof(float)));
    
    // 1. Max Pool Backward
    dim3 pool_grid(batch_size, out_channels, pool_output_height * pool_output_width);
    max_pool_backward_kernel<<<pool_grid, 1, 0, stream1>>>(
        d_grad_output,
        d_conv_grad,
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
    CHECK_LAST_CUDA_ERROR();
    
    // 2. ReLU Backward
    int block_size = 256;
    int num_blocks = (conv_size + block_size - 1) / block_size;
    
    relu_backward_kernel<<<num_blocks, block_size, 0, stream2>>>(
        d_conv_grad,
        d_conv_output_cache,
        d_conv_grad,
        conv_size
    );
    CHECK_LAST_CUDA_ERROR();
    
    // 3. Convolution Backward
    dim3 conv_grid(batch_size, out_channels, conv_output_height * conv_output_width);
    conv_backward_kernel<<<conv_grid, 1, 0, stream3>>>(
        d_conv_grad,
        d_cache,
        d_weights,
        d_grad_input,
        d_grad_weights,
        d_grad_biases,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        conv_output_height,
        conv_output_width
    );
    CHECK_LAST_CUDA_ERROR();
    
    // Update weights and biases using Adam optimizer
    weights_optimizer.update(d_weights, d_grad_weights, stream1);
    bias_optimizer.update(d_biases, d_grad_biases, stream2);
    
    // Cleanup
    cudaFree(d_grad_weights);
    cudaFree(d_grad_biases);
    cudaFree(d_conv_grad);
    
    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
}

// Destructor
ConvBlock::~ConvBlock() {
    free_memory();
    if (streams_initialized) {
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream3);
    }
}

void ConvBlock::allocate_memory(int batch_size) {
    // Free existing memory if any
    free_memory();

    // Calculate output dimensions
    conv_output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    conv_output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    pool_output_height = (conv_output_height - pool_size) / pool_stride + 1;
    pool_output_width = (conv_output_width - pool_size) / pool_stride + 1;

    // Calculate sizes
    size_t conv_size = batch_size * out_channels * conv_output_height * conv_output_width;
    size_t input_size = batch_size * in_channels * input_height * input_width;

    std::cout << "conv_size: " << conv_size << std::endl;
    std::cout << "input_size: " << input_size << std::endl;
    
    // Allocate memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv_output_cache, conv_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pool_indices, conv_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_cache, input_size * sizeof(float)));
    
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