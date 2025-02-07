#include "include/conv_block.cuh"
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>


// Performs forward convolution, applies ReLU activation
__global__ void conv_forward_kernel(
    const float* input,           // Input tensor [batch_size, in_channels, height, width]
    const float* weights,         // Weight tensor [out_channels, in_channels, kernel_size, kernel_size] 
    const float* biases,          // Bias tensor [out_channels]
    float* conv_output,           // Raw convolution output
    float* relu_output,           // Output after ReLU activation
    int batch_size,
    int in_channels, 
    int out_channels,
    int height,
    int width, 
    int kernel_size,
    int stride,
    int padding,
    int output_height,
    int output_width) {
    
        __shared__ float shared_input[TILE_SIZE + BLOCK_SIZE - 1][TILE_SIZE + BLOCK_SIZE - 1];
    __shared__ float shared_weights[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate output position
    int b = blockIdx.x;                                    // Batch index
    int oc = blockIdx.y;                                   // Output channel
    int h = (blockIdx.z / ((output_width + TILE_SIZE - 1) / TILE_SIZE)) * TILE_SIZE + threadIdx.y;
    int w = (blockIdx.z % ((output_width + TILE_SIZE - 1) / TILE_SIZE)) * TILE_SIZE + threadIdx.x;

    // Initialize accumulator
    float sum = 0.0f;
    if (threadIdx.x < BLOCK_SIZE && threadIdx.y < BLOCK_SIZE) {
        sum = biases[oc];
    }

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ic++) {
        // Load input tile into shared memory
        for (int i = threadIdx.y; i < TILE_SIZE + kernel_size - 1; i += BLOCK_SIZE) {
            for (int j = threadIdx.x; j < TILE_SIZE + kernel_size - 1; j += BLOCK_SIZE) {
                int ih = h * stride - padding + i;
                int iw = w * stride - padding + j;
                
                if (ih >= 0 && ih < height && iw >= 0 && iw < width && b < batch_size) {
                    shared_input[i][j] = input[((b * in_channels + ic) * height + ih) * width + iw];
                } else {
                    shared_input[i][j] = 0.0f;
                }
            }
        }

        // Load weights into shared memory
        if (threadIdx.y < kernel_size && threadIdx.x < kernel_size) {
            shared_weights[threadIdx.y][threadIdx.x] = 
                weights[((oc * in_channels + ic) * kernel_size + threadIdx.y) * kernel_size + threadIdx.x];
        }

        __syncthreads();

        // Compute convolution for this tile
        if (threadIdx.x < BLOCK_SIZE && threadIdx.y < BLOCK_SIZE && 
            h < output_height && w < output_width && b < batch_size && oc < out_channels) {
            
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ih = threadIdx.y * stride + kh;
                    int iw = threadIdx.x * stride + kw;
                    sum += shared_input[ih][iw] * shared_weights[kh][kw];
                }
            }
        }

        __syncthreads();
    }

    // Write output
    if (h < output_height && w < output_width && b < batch_size && oc < out_channels) {
        int output_idx = ((b * out_channels + oc) * output_height + h) * output_width + w;
        conv_output[output_idx] = sum;
        
        // ReLU activation
        const float alpha = 0.01f;  // Leaky ReLU slope
        relu_output[output_idx] = sum > 0 ? sum : alpha * sum;
    }
}

// Performs max pooling and tracks indices for backprop
__global__ void max_pool_forward_kernel(
    const float* input,           // Input tensor [batch_size, channels, height, width]
    float* output,                // Pooled output
    int* pool_indices,            // Indices of max values for backprop
    int batch_size,
    int channels,
    int height,
    int width,
    int pool_size,
    int pool_stride,
    int output_height,
    int output_width) {
    
    // Calculate output position
    int b = blockIdx.x;                                    // Batch index
    int c = blockIdx.y;                                    // Channel
    int idx = blockIdx.z * blockDim.x + threadIdx.x;
    int h = idx / output_width;                           // Output height position
    int w = idx % output_width;                           // Output width position

    // Bounds checking
    if (b >= batch_size || c >= channels || h >= output_height || w >= output_width) 
        return;
    
    // Track maximum value and its position
    float max_val = -INFINITY;
    int max_idx = -1;
    
    // Compute max over pooling window
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
                }
            }
        }
    }
    
    // Write outputs
    int output_idx = ((b * channels + c) * output_height + h) * output_width + w;
    output[output_idx] = max_val;
    pool_indices[output_idx] = max_idx;

}

// Computes gradients for convolution layer
__global__ void conv_backward_kernel(
    const float* grad_output,     // Gradient from next layer
    const float* weights,         // Layer weights
    float* grad_input,            // Gradient w.r.t input
    float* grad_weights,          // Gradient w.r.t weights  
    float* grad_biases,           // Gradient w.r.t biases
    const float* input,           // Layer input
    const float* relu_output,     // ReLU activation output
    int batch_size,
    int in_channels,
    int out_channels, 
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int output_height,
    int output_width) {
    
    // Calculate position
    int b = blockIdx.x;
    int oc = blockIdx.y;
    int h = blockIdx.z / output_width;
    int w = blockIdx.z % output_width;
    
    // Bounds checking
    if (b >= batch_size || oc >= out_channels || h >= output_height || w >= output_width) 
        return;
    
    int output_idx = ((b * out_channels + oc) * output_height + h) * output_width + w;
    float grad = grad_output[output_idx];
    
    // ReLU backward pass - zero out gradient where input was negative
    if (relu_output[output_idx] <= 0) {
        grad = 0;
    }

    // Accumulate bias gradients
    atomicAdd(&grad_biases[oc], grad);
    
    // Compute weight and input gradients
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = h * stride - padding + kh;
                int iw = w * stride - padding + kw;
                
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = ((b * in_channels + ic) * height + ih) * width + iw;
                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

                    // Accumulate gradients atomically since multiple threads write to same locations
                    atomicAdd(&grad_weights[weight_idx], input[input_idx] * grad);
                    atomicAdd(&grad_input[input_idx], weights[weight_idx] * grad);
                }
            }
        }
    }
}

__global__ void max_pool_backward_kernel(
    const float* grad_output,     // Gradient from next layer
    float* grad_input,            // Gradient to previous layer
    const int* pool_indices,      // Saved max indices from forward pass
    int batch_size,
    int channels,
    int input_height,            // Conv output height
    int input_width,             // Conv output width
    int output_height,           // Pooled output height
    int output_width             // Pooled output width
) {
    // Calculate position
    int b = blockIdx.x;                                    // Batch index
    int c = blockIdx.y;                                    // Channel
    int idx = blockIdx.z * blockDim.x + threadIdx.x;
    int h = idx / output_width;                           // Output height position
    int w = idx % output_width;                           // Output width position
    
    if (b >= batch_size || c >= channels || h >= output_height || w >= output_width) 
        return;
        
    int output_idx = ((b * channels + c) * output_height + h) * output_width + w;
    int input_idx = pool_indices[output_idx];
    
    // Propagate gradient to max element's position
    float grad = grad_output[output_idx];
    atomicAdd(&grad_input[input_idx], grad);
}

// Constructor initializes layer parameters and allocates GPU memory
ConvBlock::ConvBlock(int in_channels, int out_channels, int kernel_size, 
                     int stride, int padding, int pool_size, int pool_stride, 
                     float learning_rate)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size),
      stride(stride), padding(padding), pool_size(pool_size), 
      pool_stride(pool_stride), learning_rate(learning_rate), weights_optimizer(learning_rate),
      bias_optimizer(learning_rate),
      d_weights(nullptr), d_biases(nullptr), d_cache(nullptr),
      d_conv_output_cache(nullptr), d_relu_output_cache(nullptr),
      d_pool_indices(nullptr), current_batch_size(0) {

    std::cout << "Initializing ConvBlock with:" << std::endl;
    std::cout << "in_channels: " << in_channels << std::endl;
    std::cout << "out_channels: " << out_channels << std::endl;
    std::cout << "kernel_size: " << kernel_size << std::endl;
    std::cout << "stride: " << stride << std::endl;
    std::cout << "padding: " << padding << std::endl;
    std::cout << "pool_size: " << pool_size << std::endl;
    std::cout << "pool_stride: " << pool_stride << std::endl;
    std::cout << "learning_rate: " << learning_rate << std::endl;

    // Validate parameters
    if (kernel_size <= 0 || stride <= 0 || padding < 0 || pool_size <= 0 || pool_stride <= 0) {
        throw std::invalid_argument("Invalid convolution parameters");
    }
    
    // Initialize weights and biases on CPU
    std::vector<float> h_weights(out_channels * in_channels * kernel_size * kernel_size);
    std::vector<float> h_biases(out_channels);
    
    // Xavier/Glorot initialization for weights
    float std_dev = sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0f, std_dev);
    
    for (auto& w : h_weights) {
        w = distribution(gen);
    }
    std::fill(h_biases.begin(), h_biases.end(), 0.01f);

    // Initialize optimizers
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
}

// Destructor frees GPU memory
ConvBlock::~ConvBlock() {
    std::cout << "Destroying ConvBlock" << std::endl;
    free_memory();
}

// Allocates GPU memory for intermediate results
void ConvBlock::allocate_memory(int batch_size) {
    // Free any existing allocations
    if (d_conv_output_cache) cudaFree(d_conv_output_cache);
    if (d_relu_output_cache) cudaFree(d_relu_output_cache);
    if (d_pool_indices) cudaFree(d_pool_indices);
    if (d_cache) cudaFree(d_cache);

    d_cache = nullptr;
    d_conv_output_cache = nullptr;
    d_relu_output_cache = nullptr;
    d_pool_indices = nullptr;

    // Calculate output dimensions
    conv_output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    conv_output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    pool_output_height = (conv_output_height - pool_size) / pool_stride + 1;
    pool_output_width = (conv_output_width - pool_size) / pool_stride + 1;

    size_t conv_size = batch_size * out_channels * conv_output_height * conv_output_width;
    size_t input_size = batch_size * in_channels * input_height * input_width;

    // Allocate memory for intermediate results
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv_output_cache, conv_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_relu_output_cache, conv_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pool_indices, conv_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_cache, input_size * sizeof(float)));

    current_batch_size = batch_size;
}

// Frees all GPU memory
void ConvBlock::free_memory() {
    if (d_weights) {
        cudaFree(d_weights);
        d_weights = nullptr;
    }
    if (d_biases) {
        cudaFree(d_biases);
        d_biases = nullptr;
    }
    if (d_cache) {
        cudaFree(d_cache);
        d_cache = nullptr;
    }
    if (d_conv_output_cache) {
        cudaFree(d_conv_output_cache);
        d_conv_output_cache = nullptr;
    }
    if (d_relu_output_cache) {
        cudaFree(d_relu_output_cache);
        d_relu_output_cache = nullptr;
    }
    if (d_pool_indices) {
        cudaFree(d_pool_indices);
        d_pool_indices = nullptr;
    }
}

// Forward pass: convolution -> ReLU -> max pooling
void ConvBlock::forward(const float* d_input, float* d_output, int batch_size, int height, int width) {

    input_height = height;
    input_width = width;
    current_batch_size = batch_size;
    
    // Allocate memory for this forward pass
    allocate_memory(batch_size);

    // Cache input for backward pass
    size_t input_size = batch_size * in_channels * height * width * sizeof(float);
    CHECK_CUDA_ERROR(cudaMemcpy(d_cache, d_input, input_size, cudaMemcpyDeviceToDevice));

    // Launch convolution kernel
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(batch_size,
              out_channels,
              ((output_height + TILE_SIZE - 1) / TILE_SIZE) * 
              ((output_width + TILE_SIZE - 1) / TILE_SIZE));

    conv_forward_kernel<<<gridDim, blockDim>>>(
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
    CHECK_LAST_CUDA_ERROR();
    
    // Synchronize and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in conv forward: %s\n", cudaGetErrorString(err));
        return;
    }

    // Launch pooling kernel
    dim3 gridDimPooling(batch_size, 
                 out_channels, 
                 (pool_output_height * pool_output_width + 255) / 256);
    dim3 blockDimPooling(256);

    max_pool_forward_kernel<<<gridDimPooling, blockDimPooling>>>(
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
    CHECK_LAST_CUDA_ERROR();
    
    // Synchronize and check for errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in pooling forward: %s\n", cudaGetErrorString(err));
        return;
    }
}

// Backward pass: computes gradients and updates parameters
void ConvBlock::backward(const float* d_grad_output, float* d_grad_input, int batch_size) {

    if (batch_size != current_batch_size) {
        throw std::invalid_argument("Batch size mismatch between forward and backward passes");
    }
    
    // Calculate sizes
    size_t weight_size = out_channels * in_channels * kernel_size * kernel_size;
    size_t bias_size = out_channels;
    size_t input_size = batch_size * in_channels * input_height * input_width;
 
    // Allocate temporary gradient buffers
    float *d_grad_weights, *d_grad_biases;
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_weights, weight_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_biases, bias_size * sizeof(float)));
    
    // Zero out gradients
    CHECK_CUDA_ERROR(cudaMemset(d_grad_weights, 0, weight_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_biases, 0, bias_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_input, 0, input_size * sizeof(float)));

    float* d_unpooled_grad;
    size_t conv_output_size = batch_size * out_channels * conv_output_height * conv_output_width;
    CHECK_CUDA_ERROR(cudaMalloc(&d_unpooled_grad, conv_output_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_unpooled_grad, 0, conv_output_size * sizeof(float)));
    
    dim3 gridDimPool(batch_size, 
                    out_channels, 
                    (pool_output_height * pool_output_width + 255) / 256);
    dim3 blockDimPool(256);
    
    max_pool_backward_kernel<<<gridDimPool, blockDimPool>>>(
        d_grad_output,
        d_unpooled_grad,
        d_pool_indices,
        batch_size,
        out_channels,
        conv_output_height,
        conv_output_width,
        pool_output_height,
        pool_output_width
    );
    CHECK_LAST_CUDA_ERROR();
 
    
    // Launch backward kernel
    int total_spatial_elements = conv_output_height * conv_output_width;
    dim3 gridDim(batch_size, out_channels, (total_spatial_elements + 255) / 256);
    dim3 blockDim(256);

    conv_backward_kernel<<<gridDim, blockDim>>>(
        d_unpooled_grad, d_weights,
        d_grad_input, d_grad_weights, d_grad_biases,
        d_cache, d_relu_output_cache,
        batch_size, in_channels, out_channels,
        input_height, input_width, kernel_size, stride, padding,
        conv_output_height, conv_output_width
    );

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in backward pass: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Update parameters using optimizers
    weights_optimizer.update(d_weights, d_grad_weights);
    bias_optimizer.update(d_biases, d_grad_biases);
    
    // Free temporary buffers
    CHECK_CUDA_ERROR(cudaFree(d_grad_weights));
    CHECK_CUDA_ERROR(cudaFree(d_grad_biases));
    CHECK_CUDA_ERROR(cudaFree(d_unpooled_grad));
}