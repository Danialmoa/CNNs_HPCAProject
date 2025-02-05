#include "include/conv_block.cuh"
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>

#define DEBUG_PRINT 1

// CUDA kernels
__global__ void conv_forward_kernel(
    const float* input, const float* weights, const float* biases,
    float* conv_output, float* relu_output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int stride, int padding,
    int output_height, int output_width) {
    
    int b = blockIdx.x;
    int oc = blockIdx.y;
    int h = blockIdx.z / output_width;
    int w = blockIdx.z % output_width;
    
    if (b >= batch_size || oc >= out_channels || h >= output_height || w >= output_width) 
        return;
    
    float sum = biases[oc];
    
    #ifdef DEBUG_PRINT
    if (b == 0 && oc == 0 && h == 0 && w == 0) {
        printf("Conv Forward - First element computation:\n");
        printf("Bias: %f\n", biases[oc]);
    }
    #endif
    
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = h * stride - padding + kh;
                int iw = w * stride - padding + kw;
                
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = ((b * in_channels + ic) * height + ih) * width + iw;
                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weights[weight_idx];
                    
                    #ifdef DEBUG_PRINT
                    if (b == 0 && oc == 0 && h == 0 && w == 0 && ic == 0 && kh == 0 && kw == 0) {
                        printf("First multiplication: input[%d]=%f * weight[%d]=%f\n", 
                               input_idx, input[input_idx], weight_idx, weights[weight_idx]);
                    }
                    #endif
                }
            }
        }
    }
    
    int output_idx = ((b * out_channels + oc) * output_height + h) * output_width + w;
    conv_output[output_idx] = sum;
    relu_output[output_idx] = fmaxf(0.0f, sum);
    
    #ifdef DEBUG_PRINT
    if (b == 0 && oc == 0 && h == 0 && w == 0) {
        printf("First output: conv=%f, relu=%f\n", conv_output[output_idx], relu_output[output_idx]);
    }
    #endif
}

__global__ void max_pool_forward_kernel(
    const float* input, float* output, int* pool_indices,
    int batch_size, int channels, int height, int width,
    int pool_size, int pool_stride,
    int output_height, int output_width) {
    
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = (blockIdx.z * blockDim.x + threadIdx.x) / output_width;
    int w = (blockIdx.z * blockDim.x + threadIdx.x) % output_width;
    
    if (b >= batch_size || c >= channels || h >= output_height || w >= output_width) 
        return;
    
    float max_val = -INFINITY;
    int max_idx = -1;
    
    #ifdef DEBUG_PRINT
    if (b == 0 && c == 0 && h == 0 && w == 0) {
        printf("MaxPool Forward - Processing first window:\n");
    }
    #endif
    
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int ih = h * pool_stride + ph;
            int iw = w * pool_stride + pw;
            
            if (ih < height && iw < width) {
                int idx = ((b * channels + c) * height + ih) * width + iw;
                float val = input[idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = idx;
                    
                    #ifdef DEBUG_PRINT
                    if (b == 0 && c == 0 && h == 0 && w == 0) {
                        printf("New max found: val=%f at idx=%d\n", val, idx);
                    }
                    #endif
                }
            }
        }
    }
    
    int output_idx = ((b * channels + c) * output_height + h) * output_width + w;
    output[output_idx] = max_val;
    pool_indices[output_idx] = max_idx;
    
    #ifdef DEBUG_PRINT
    if (b == 0 && c == 0 && h == 0 && w == 0) {
        printf("First pool output: val=%f, index=%d\n", max_val, max_idx);
    }
    #endif
}

__global__ void conv_backward_kernel(
    const float* grad_output, const float* weights,
    float* grad_input, float* grad_weights, float* grad_biases,
    const float* input, const float* relu_output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int stride, int padding,
    int output_height, int output_width) {
    
    int b = blockIdx.x;
    int oc = blockIdx.y;
    int h = blockIdx.z / output_width;
    int w = blockIdx.z % output_width;
    
    if (b >= batch_size || oc >= out_channels || h >= output_height || w >= output_width) 
        return;
    
    int output_idx = ((b * out_channels + oc) * output_height + h) * output_width + w;
    float grad = grad_output[output_idx];
    
    #ifdef DEBUG_PRINT
    if (b == 0 && oc == 0 && h == 0 && w == 0) {
        printf("Conv Backward - Initial gradient: %f\n", grad);
    }
    #endif
    
    // ReLU backward
    if (relu_output[output_idx] <= 0) {
        grad = 0;
        #ifdef DEBUG_PRINT
        if (b == 0 && oc == 0 && h == 0 && w == 0) {
            printf("ReLU backward: gradient zeroed\n");
        }
        #endif
    }
    
    // Bias gradient
    atomicAdd(&grad_biases[oc], grad);
    
    #ifdef DEBUG_PRINT
    if (b == 0 && oc == 0 && h == 0 && w == 0) {
        printf("Updated bias gradient for channel %d\n", oc);
    }
    #endif
    
    // Weight and input gradients
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = h * stride - padding + kh;
                int iw = w * stride - padding + kw;
                
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = ((b * in_channels + ic) * height + ih) * width + iw;
                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    
                    atomicAdd(&grad_weights[weight_idx], input[input_idx] * grad);
                    atomicAdd(&grad_input[input_idx], weights[weight_idx] * grad);
                    
                    #ifdef DEBUG_PRINT
                    if (b == 0 && oc == 0 && ic == 0 && kh == 0 && kw == 0) {
                        printf("First gradient update: weight[%d]+=(%f*%f), input[%d]+=(%f*%f)\n",
                               weight_idx, input[input_idx], grad,
                               input_idx, weights[weight_idx], grad);
                    }
                    #endif
                }
            }
        }
    }
}

ConvBlock::ConvBlock(int in_channels, int out_channels, int kernel_size, 
                     int stride, int padding, int pool_size, int pool_stride, 
                     float learning_rate)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size),
      stride(stride), padding(padding), pool_size(pool_size), 
      pool_stride(pool_stride), learning_rate(learning_rate), weights_optimizer(learning_rate),
      bias_optimizer(learning_rate),
      d_weights(nullptr), d_biases(nullptr), d_cache(nullptr),
      d_conv_output_cache(nullptr), d_relu_output_cache(nullptr),
      d_pool_indices(nullptr) {

    std::cout << "Initializing ConvBlock with:" << std::endl
              << "  in_channels: " << in_channels << std::endl
              << "  out_channels: " << out_channels << std::endl
              << "  kernel_size: " << kernel_size << std::endl
              << "  stride: " << stride << std::endl
              << "  padding: " << padding << std::endl
              << "  pool_size: " << pool_size << std::endl
              << "  pool_stride: " << pool_stride << std::endl
              << "  learning_rate: " << learning_rate << std::endl;

    if (kernel_size <= 0 || stride <= 0 || padding < 0 || pool_size <= 0 || pool_stride <= 0) {
        throw std::invalid_argument("Invalid convolution parameters");
    }
    
    // Initialize weights and biases
    std::vector<float> h_weights(out_channels * in_channels * kernel_size * kernel_size);
    std::vector<float> h_biases(out_channels);
    
    // Xavier initialization
    float std_dev = sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0f, std_dev);
    
    for (auto& w : h_weights) {
        w = distribution(gen);
    }
    std::fill(h_biases.begin(), h_biases.end(), 0.01f);

    std::cout << "Initialized weights with std_dev: " << std_dev << std::endl;

    weights_optimizer.init(out_channels * in_channels * kernel_size * kernel_size);
    bias_optimizer.init(out_channels);
    
    // Allocate and copy weights and biases to GPU
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights, h_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_biases, h_biases.size() * sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, h_weights.data(), 
                               h_weights.size() * sizeof(float), 
                               cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_biases, h_biases.data(), 
                               h_biases.size() * sizeof(float), 
                               cudaMemcpyHostToDevice));
    
    std::cout << "Successfully allocated and copied weights and biases to GPU" << std::endl;
}

ConvBlock::~ConvBlock() {
    std::cout << "Destroying ConvBlock" << std::endl;
    free_memory();
}

void ConvBlock::allocate_memory(int batch_size) {
    free_memory();
    // Calculate output dimensions
    conv_output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    conv_output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    pool_output_height = (conv_output_height - pool_size) / pool_stride + 1;
    pool_output_width = (conv_output_width - pool_size) / pool_stride + 1;
    
    std::cout << "Allocating memory for batch_size: " << batch_size << std::endl
              << "Conv output dimensions: " << conv_output_height << "x" << conv_output_width << std::endl
              << "Pool output dimensions: " << pool_output_height << "x" << pool_output_width << std::endl;
    
    size_t conv_size = batch_size * out_channels * conv_output_height * conv_output_width;
    size_t input_size = batch_size * in_channels * input_height * input_width;
    
    // Allocate memory for intermediate results
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv_output_cache, conv_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_relu_output_cache, conv_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pool_indices, conv_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_cache, input_size * sizeof(float)));
    
    std::cout << "Successfully allocated GPU memory" << std::endl;
}

void ConvBlock::free_memory() {
    std::cout << "Freeing ConvBlock GPU memory" << std::endl;
    if (d_weights) cudaFree(d_weights);
    if (d_biases) cudaFree(d_biases);
    if (d_cache) cudaFree(d_cache);
    if (d_conv_output_cache) cudaFree(d_conv_output_cache);
    if (d_relu_output_cache) cudaFree(d_relu_output_cache);
    if (d_pool_indices) cudaFree(d_pool_indices);
}

void ConvBlock::forward(const float* d_input, float* d_output, int batch_size, int height, int width) {
    input_height = height;
    input_width = width;
    
    // Allocate memory for this forward pass
    allocate_memory(batch_size);

    // Copy input to cache
    size_t input_size = batch_size * in_channels * input_height * input_width;
    cudaMemcpy(d_cache, d_input, input_size * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Very basic kernel launch configuration
    int threadsPerBlock = 256;
    int totalElements = batch_size * out_channels * conv_output_height * conv_output_width;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch convolution kernel
    conv_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_cache,
        d_weights,
        d_biases,
        d_conv_output_cache,
        d_relu_output_cache,
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
    
    // Synchronize and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Launch pooling kernel
    totalElements = batch_size * out_channels * pool_output_height * pool_output_width;
    blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    
    max_pool_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_relu_output_cache,
        d_output,
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
    
    // Synchronize and check for errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }
}

void ConvBlock::backward(const float* d_grad_output, float* d_grad_input, int batch_size) {
    std::cout << "Backward pass - batch_size: " << batch_size << std::endl;
    
    // Allocate memory for gradients
    float *d_grad_weights, *d_grad_biases;
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_weights, 
        out_channels * in_channels * kernel_size * kernel_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_biases, out_channels * sizeof(float)));
    
    // Zero out gradients
    CHECK_CUDA_ERROR(cudaMemset(d_grad_weights, 0, 
        out_channels * in_channels * kernel_size * kernel_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_biases, 0, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_input, 0, 
        batch_size * in_channels * input_height * input_width * sizeof(float)));
    
    // Simplified kernel launch configuration
    const int THREADS_PER_BLOCK = 256;
    int total_elements = batch_size * out_channels * conv_output_height * conv_output_width;
    int num_blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    std::cout << "Launching conv_backward_kernel with " << num_blocks 
              << " blocks, " << THREADS_PER_BLOCK << " threads per block" << std::endl;
    
    conv_backward_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_grad_output, d_weights,
        d_grad_input, d_grad_weights, d_grad_biases,
        d_cache, d_relu_output_cache,
        batch_size, in_channels, out_channels,
        input_height, input_width, kernel_size, stride, padding,
        conv_output_height, conv_output_width
    );
    CHECK_LAST_CUDA_ERROR();
    
    // Update weights and biases using optimizers
    weights_optimizer.update(d_weights, d_grad_weights);
    bias_optimizer.update(d_biases, d_grad_biases);
    
    // Free temporary memory
    cudaFree(d_grad_weights);
    cudaFree(d_grad_biases);
    
    std::cout << "Backward pass completed successfully" << std::endl;
}