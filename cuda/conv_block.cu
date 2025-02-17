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
    float grad = grad_output[out_idx] / batch_size;
    
    // Clip gradients 
    const float CLIP_VALUE = 5.0f;
    grad = fmaxf(fminf(grad, CLIP_VALUE), -CLIP_VALUE);

    // Accumulate bias gradients
    atomicAdd(&grad_biases[out_ch], grad);

    // L2 regularization coefficient
    const float l2_reg = 0.0001f;
    
    // Compute gradients for weights and input
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h * stride - padding + kh;
                int w_in = w * stride - padding + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int in_idx = ((batch * in_channels + in_ch) * height + h_in) * width + w_in;
                    int weight_idx = ((out_ch * in_channels + in_ch) * kernel_size + kh) * kernel_size + kw;
                    float weight_grad = input[in_idx] * grad + l2_reg * weights[weight_idx];
                    // Gradient w.r.t. weights
                    atomicAdd(&grad_weights[weight_idx], weight_grad);
                    
                    // Gradient w.r.t. input
                    atomicAdd(&grad_input[in_idx], weights[weight_idx] * grad);
                }
            }
        }
    }
}

// Add new kernel for batch normalization
__global__ void batch_norm_forward_kernel(
    float* input,           // [batch, channels, height, width]
    const float* gamma,     // [channels]
    const float* beta,      // [channels]
    float* running_mean,    // [channels]
    float* running_var,     // [channels]
    float* batch_mean,      // [channels]
    float* batch_var,       // [channels]
    int batch_size, int channels, int height, int width,
    float epsilon, float momentum,
    bool is_training
) {
    const int c = blockIdx.x;
    if (c >= channels) return;

    float mean = 0.0f;
    float m2 = 0.0f;
    const int spatial_size = height * width;
    const int N = batch_size * spatial_size;

    // Step 1: Calculate mean
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = ((b * channels + c) * height + h) * width + w;
                mean += input[idx];
            }
        }
    }
    mean /= N;
    
    // Step 2: Calculate variance
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = ((b * channels + c) * height + h) * width + w;
                float diff = input[idx] - mean;
                m2 += diff * diff;
            }
        }
    }
    float var = m2 / N;

    if (is_training) {
        batch_mean[c] = mean;
        batch_var[c] = var;
        
        // Update running statistics
        running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * mean;
        running_var[c] = momentum * running_var[c] + (1.0f - momentum) * var;
    } else {
        mean = running_mean[c];
        var = running_var[c];
    }

    // Step 3: Normalize and scale
    float std = sqrtf(var + epsilon);
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = ((b * channels + c) * height + h) * width + w;
                input[idx] = gamma[c] * (input[idx] - mean) / std + beta[c];
            }
        }
    }
}

// Add batch norm backward kernel
__global__ void batch_norm_backward_kernel(
    const float* grad_output,     // [batch, channels, height, width]
    const float* input,           // [batch, channels, height, width]
    const float* gamma,           // [channels]
    const float* batch_mean,      // [channels]
    const float* batch_var,       // [channels]
    float* grad_input,            // [batch, channels, height, width]
    float* grad_gamma,            // [channels]
    float* grad_beta,             // [channels]
    int batch_size, int channels, int height, int width,
    float epsilon
) {
    const int c = blockIdx.x;
    if (c >= channels) return;

    const int spatial_size = height * width;
    const int N = batch_size * spatial_size;
    const float std = sqrtf(batch_var[c] + epsilon);
    
    float sum_grad = 0.0f;
    float sum_grad_h = 0.0f;
    float sum_grad_h_x = 0.0f;

    // First pass: compute sums for gradient calculations
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                const int idx = ((b * channels + c) * height + h) * width + w;
                const float x = input[idx];
                const float dout = grad_output[idx];
                
                sum_grad += dout;
                sum_grad_h_x += dout * (x - batch_mean[c]);
            }
        }
    }

    // Compute gradients for gamma and beta
    atomicAdd(&grad_beta[c], sum_grad);
    atomicAdd(&grad_gamma[c], sum_grad_h_x / std);

    // Second pass: compute gradient for input
    const float mean = batch_mean[c];
    const float var = batch_var[c];
    const float inv_std = 1.0f / std;
    const float inv_N = 1.0f / N;

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                const int idx = ((b * channels + c) * height + h) * width + w;
                const float x = input[idx];
                const float dout = grad_output[idx];
                
                // Gradient with respect to input
                const float dx_normalized = dout * gamma[c];
                const float dx = inv_std * (
                    dx_normalized - 
                    (sum_grad * inv_N) - 
                    ((x - mean) / var * sum_grad_h_x * inv_N)
                );
                
                grad_input[idx] = dx;
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
      d_cache(nullptr), d_conv_output_cache(nullptr), d_pool_indices(nullptr),
      d_weights(nullptr), d_biases(nullptr),
      weights_optimizer(lr), bias_optimizer(lr),
      gamma_optimizer(lr), beta_optimizer(lr) {
    
    init_weights_and_optimizers();

    // Initialize batch norm parameters
    CHECK_CUDA_ERROR(cudaMalloc(&d_gamma, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_beta, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_running_mean, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_running_var, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_mean, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_var, out_channels * sizeof(float)));

    // Initialize values
    std::vector<float> h_gamma(out_channels, 1.0f);
    std::vector<float> h_beta(out_channels, 0.0f);
    std::vector<float> h_zeros(out_channels, 0.0f);
    std::vector<float> h_ones(out_channels, 1.0f);

    CHECK_CUDA_ERROR(cudaMemcpy(d_gamma, h_gamma.data(), out_channels * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_beta, h_beta.data(), out_channels * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_running_mean, h_zeros.data(), out_channels * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_running_var, h_ones.data(), out_channels * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize optimizers for batch norm parameters
    gamma_optimizer.init(out_channels);
    beta_optimizer.init(out_channels);
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
    // Initialize streams if not already done
    if (!streams_initialized) {
        init_streams();
    }

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
    size_t input_size = batch_size * in_channels * height * width * sizeof(float);
    CHECK_CUDA_ERROR(cudaMemcpy(d_cache, d_input, input_size, cudaMemcpyDeviceToDevice));

    // Ensure batch_size is divisible by number of streams
    int streams_count = 3;
    int stream_batch_size = (batch_size + streams_count - 1) / streams_count;

    dim3 threadsPerBlock(16, 16);
    
    // Calculate shared memory size
    size_t shared_mem_size = (
        (threadsPerBlock.x + kernel_size - 1) * (threadsPerBlock.y + kernel_size - 1) +
        kernel_size * kernel_size
    ) * sizeof(float);

    // Check shared memory size against device limit
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (shared_mem_size > prop.sharedMemPerBlock) {
        throw std::runtime_error("Shared memory size exceeds device limit");
    }

    // Process batches in streams
    for (int i = 0; i < streams_count; i++) {
        int start_batch = i * stream_batch_size;
        int current_batch_size = std::min(stream_batch_size, batch_size - start_batch);
        
        if (current_batch_size <= 0) continue;

        dim3 numBlocks(
            (conv_output_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (conv_output_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
            current_batch_size * out_channels
        );

        cudaStream_t current_stream = (i == 0) ? stream1 : (i == 1) ? stream2 : stream3;

        conv_forward_kernel<<<numBlocks, threadsPerBlock, shared_mem_size, current_stream>>>(
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
        CHECK_LAST_CUDA_ERROR();
    }

    // Synchronize after convolution
    cudaDeviceSynchronize();

    // After convolution but before ReLU, add batch normalization
    batch_norm_forward_kernel<<<out_channels, 1>>>(
        d_conv_output_cache,  // in-place normalization
        d_gamma,
        d_beta,
        d_running_mean,
        d_running_var,
        d_batch_mean,
        d_batch_var,
        batch_size,
        out_channels,
        conv_output_height,
        conv_output_width,
        bn_epsilon,
        bn_momentum,
        is_training
    );
    CHECK_LAST_CUDA_ERROR();

    // ReLU and pooling can be done on the entire batch at once
    const int total_conv_elements = batch_size * out_channels * conv_output_height * conv_output_width;
    const int relu_threads = 256;
    const int relu_blocks = (total_conv_elements + relu_threads - 1) / relu_threads;
    
    relu_kernel<<<relu_blocks, relu_threads>>>(
        d_conv_output_cache,
        total_conv_elements
    );
    CHECK_LAST_CUDA_ERROR();

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
    // Allocate temporary gradients for batch norm parameters
    float *d_grad_gamma, *d_grad_beta, *d_grad_weights, *d_grad_biases;
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_gamma, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_beta, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_weights, 
        out_channels * in_channels * kernel_size * kernel_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_biases, out_channels * sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaMemset(d_grad_gamma, 0, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_beta, 0, out_channels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_weights, 0, 
        out_channels * in_channels * kernel_size * kernel_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_biases, 0, out_channels * sizeof(float)));

    // Temporary buffer for gradients before batch norm
    float* d_grad_pre_bn;
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_pre_bn, 
        batch_size * out_channels * conv_output_height * conv_output_width * sizeof(float)));

    // Define block and grid dimensions for ReLU backward
    const int block_size = 256;
    const int total_conv_elements = batch_size * out_channels * conv_output_height * conv_output_width;
    const int num_blocks = (total_conv_elements + block_size - 1) / block_size;

    // Backward pass through ReLU
    relu_backward_kernel<<<num_blocks, block_size>>>(
        d_grad_output,
        d_conv_output_cache,
        d_grad_pre_bn,
        total_conv_elements
    );
    CHECK_LAST_CUDA_ERROR();

    // Backward pass through batch normalization
    batch_norm_backward_kernel<<<out_channels, 1>>>(
        d_grad_pre_bn,
        d_conv_output_cache,
        d_gamma,
        d_batch_mean,
        d_batch_var,
        d_grad_pre_bn,
        d_grad_gamma,
        d_grad_beta,
        batch_size,
        out_channels,
        conv_output_height,
        conv_output_width,
        bn_epsilon
    );
    CHECK_LAST_CUDA_ERROR();

    // Define grid dimensions for convolution backward
    dim3 conv_grid(batch_size, out_channels, conv_output_height * conv_output_width);

    // Update batch norm parameters using Adam optimizer
    gamma_optimizer.update(d_gamma, d_grad_gamma, stream1);
    beta_optimizer.update(d_beta, d_grad_beta, stream2);

    // Continue with convolution backward pass
    conv_backward_kernel<<<conv_grid, 1, 0, stream3>>>(
        d_grad_pre_bn,
        d_cache,
        d_weights,
        d_grad_input,
        d_grad_weights,
        d_grad_biases,
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

    // Update weights and biases using Adam optimizer
    weights_optimizer.update(d_weights, d_grad_weights, stream1);
    bias_optimizer.update(d_biases, d_grad_biases, stream2);

    // Clean up temporary memory
    cudaFree(d_grad_gamma);
    cudaFree(d_grad_beta);
    cudaFree(d_grad_weights);
    cudaFree(d_grad_biases);
    cudaFree(d_grad_pre_bn);

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
    if (d_gamma) cudaFree(d_gamma);
    if (d_beta) cudaFree(d_beta);
    if (d_running_mean) cudaFree(d_running_mean);
    if (d_running_var) cudaFree(d_running_var);
    if (d_batch_mean) cudaFree(d_batch_mean);
    if (d_batch_var) cudaFree(d_batch_var);
}

void ConvBlock::allocate_memory(int batch_size) {
    // Synchronize before freeing memory
    cudaDeviceSynchronize();
    free_memory();
    
    // Calculate output dimensions
    conv_output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    conv_output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    pool_output_height = (conv_output_height - pool_size) / pool_stride + 1;
    pool_output_width = (conv_output_width - pool_size) / pool_stride + 1;

    // Calculate sizes
    size_t input_size = batch_size * in_channels * input_height * input_width;
    size_t conv_size = batch_size * out_channels * conv_output_height * conv_output_width;
    size_t pool_indices_size = batch_size * out_channels * pool_output_height * pool_output_width;

    // Allocate memory with error checking
    cudaError_t err;
    
    err = cudaMalloc(&d_cache, input_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_cache");
    
    err = cudaMalloc(&d_conv_output_cache, conv_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_conv_output_cache");
    
    err = cudaMalloc(&d_pool_indices, pool_indices_size * sizeof(int));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_pool_indices");

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