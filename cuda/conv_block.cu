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
    extern __shared__ float shared_mem[];
    
    // Calculate thread and block indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    const int out_ch = blockIdx.z % out_channels;
    const int batch = blockIdx.z / out_channels;
    
    if (x >= out_width || y >= out_height || out_ch >= out_channels || batch >= batch_size) return;
    
    // Calculate shared memory layout
    float* shared_input = shared_mem;
    float* shared_weights = shared_input + (blockDim.y + kernel_size - 1) * (blockDim.x + kernel_size - 1);
    
    float sum = biases[out_ch];
    
    // Load input tiles into shared memory
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        // Load input patch into shared memory
        const int tile_h = blockDim.y + kernel_size - 1;
        const int tile_w = blockDim.x + kernel_size - 1;
        
        for (int i = ty; i < tile_h; i += blockDim.y) {
            for (int j = tx; j < tile_w; j += blockDim.x) {
                int h_in = blockIdx.y * blockDim.y + i - padding;
                int w_in = blockIdx.x * blockDim.x + j - padding;
                
                float val = 0.0f;
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    val = input[((batch * in_channels + in_ch) * height + h_in) * width + w_in];
                }
                shared_input[i * tile_w + j] = val;
            }
        }
        
        // Load weights into shared memory
        for (int i = ty; i < kernel_size; i += blockDim.y) {
            for (int j = tx; j < kernel_size; j += blockDim.x) {
                if (i < kernel_size && j < kernel_size) {
                    shared_weights[i * kernel_size + j] = 
                        weights[((out_ch * in_channels + in_ch) * kernel_size + i) * kernel_size + j];
                }
            }
        }
        
        __syncthreads();
        
        // Compute convolution using shared memory
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_offset = ty * stride + kh;
                int w_offset = tx * stride + kw;
                sum += shared_input[h_offset * (blockDim.x + kernel_size - 1) + w_offset] * 
                       shared_weights[kh * kernel_size + kw];
            }
        }
        
        __syncthreads();
    }
    
    output[((batch * out_channels + out_ch) * out_height + y) * out_width + x] = sum;
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
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int ch = blockIdx.z % channels;
    const int batch = blockIdx.z / channels;
    
    if (x >= out_width || y >= out_height || ch >= channels || batch >= batch_size) return;
    
    float max_val = -INFINITY;
    int max_idx = -1;
    
    // Compute the input window position
    const int h_start = y * stride;
    const int w_start = x * stride;
    
    // Find maximum in the pool window
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            const int h_in = h_start + ph;
            const int w_in = w_start + pw;
            
            if (h_in < height && w_in < width) {
                const int in_idx = ((batch * channels + ch) * height + h_in) * width + w_in;
                const float val = input[in_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = in_idx;
                }
            }
        }
    }
    
    // Write output
    const int out_idx = ((batch * channels + ch) * out_height + y) * out_width + x;
    output[out_idx] = max_val;
    indices[out_idx] = max_idx;
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
    
    // Initialize pointers to nullptr
    d_cache = nullptr;
    d_conv_output_cache = nullptr;
    d_pool_indices = nullptr;
    d_weights = nullptr;
    d_biases = nullptr;

        
    init_weights_and_optimizers();
}
void ConvBlock::init_weights_and_optimizers() {
    // Calculate sizes
    size_t weights_size = out_channels * in_channels * kernel_size * kernel_size;
    size_t bias_size = out_channels;

    // Initialize weights and biases
    std::vector<float> h_weights(weights_size);
    std::vector<float> h_biases(bias_size, 0.0f);
    
    // Xavier initialization
    float std_dev = sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (float& w : h_weights) {
        w = dist(gen);
    }
    
    // Allocate and copy weights and biases to GPU
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights, weights_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_biases, bias_size * sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, h_weights.data(), 
                               weights_size * sizeof(float), 
                               cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_biases, h_biases.data(), 
                               bias_size * sizeof(float), 
                               cudaMemcpyHostToDevice));

    // Initialize Adam optimizers with correct sizes
    weights_optimizer.init(weights_size);
    bias_optimizer.init(bias_size);
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

    // Add error checking
    if (d_cache == nullptr) {
        throw std::runtime_error("d_cache is null. Memory allocation failed.");
    }

    // Cache input for backward pass
    size_t input_size = batch_size * in_channels * height * width * sizeof(float);
    CHECK_CUDA_ERROR(cudaMemcpy(d_cache, d_input, input_size, cudaMemcpyDeviceToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (conv_output_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (conv_output_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
        out_channels * batch_size
    );
    
    // Calculate shared memory size
    size_t shared_mem_size = (
        // Input tile size
        (threadsPerBlock.x + kernel_size - 1) * (threadsPerBlock.y + kernel_size - 1) +
        // Weight tile size
        kernel_size * kernel_size
    ) * sizeof(float);

    // Split the work across streams
    int stream_batch_size = batch_size / 3;
    for (int i = 0; i < 3; i++) {
        int start_batch = i * stream_batch_size;
        int current_batch_size = (i == 2) ? batch_size - start_batch : stream_batch_size;
        
        dim3 stream_blocks(
            numBlocks.x,
            numBlocks.y,
            out_channels * current_batch_size
        );
        
        cudaStream_t current_stream = (i == 0) ? stream1 : (i == 1) ? stream2 : stream3;
        
        conv_forward_kernel<<<stream_blocks, threadsPerBlock, shared_mem_size, current_stream>>>(
            d_input + start_batch * in_channels * height * width,
            d_weights,
            d_biases,
            d_conv_output_cache + start_batch * out_channels * conv_output_height * conv_output_width,
            current_batch_size,
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
    }
    
    // 2. ReLU
    const int total_conv_elements = batch_size * out_channels * conv_output_height * conv_output_width;
    const int relu_threads = 256;
    const int relu_blocks = (total_conv_elements + relu_threads - 1) / relu_threads;
    
    relu_kernel<<<relu_blocks, relu_threads>>>(
        d_conv_output_cache, total_conv_elements
    );
    CHECK_LAST_CUDA_ERROR();
    
    // 3. Max Pooling
    dim3 poolThreads(16, 16);
    dim3 poolBlocks(
        (pool_output_width + poolThreads.x - 1) / poolThreads.x,
        (pool_output_height + poolThreads.y - 1) / poolThreads.y,
        batch_size * out_channels
    );
    
    max_pool_kernel<<<poolBlocks, poolThreads>>>(
        d_conv_output_cache,
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
}

// Backward pass implementation
void ConvBlock::backward(const float* d_grad_output, float* d_grad_input, 
                        int batch_size, int height, int width) {
    // Set dimensions if not already set
    input_height = height;
    input_width = width;
    
    // Add check for input dimensions
    if (input_height == 0 || input_width == 0) {
        throw std::runtime_error("Input dimensions not set. Call forward() before backward()");
    }
    if (batch_size != current_batch_size) {
        allocate_memory(batch_size);
    }

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
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // Calculate output dimensions
    conv_output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    conv_output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    pool_output_height = (conv_output_height - pool_size) / pool_stride + 1;
    pool_output_width = (conv_output_width - pool_size) / pool_stride + 1;

    // Calculate sizes
    size_t input_size = batch_size * in_channels * input_height * input_width;
    size_t conv_size = batch_size * out_channels * conv_output_height * conv_output_width;
    size_t pool_size = batch_size * out_channels * pool_output_height * pool_output_width;

    // Allocate memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_cache, input_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv_output_cache, conv_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pool_indices, pool_size * sizeof(int)));
    
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