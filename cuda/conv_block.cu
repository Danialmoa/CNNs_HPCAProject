#include "include/conv_block.cuh"
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>


#define TILE_SIZE 16
#define BLOCK_SIZE 16
#define MAX_THREADS 256

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
    
    __shared__ float s_grad[TILE_SIZE][TILE_SIZE];
    __shared__ float s_input[TILE_SIZE + BLOCK_SIZE - 1][TILE_SIZE + BLOCK_SIZE - 1];
    __shared__ float s_weights[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_partial_grad_weights[BLOCK_SIZE][BLOCK_SIZE];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block indices
    int b = blockIdx.x;  // batch
    int oc = blockIdx.y; // output channel
    int tile_idx = blockIdx.z;
    
    // Calculate tile position
    int tile_h = (tile_idx / ((width + TILE_SIZE - 1) / TILE_SIZE)) * TILE_SIZE;
    int tile_w = (tile_idx % ((width + TILE_SIZE - 1) / TILE_SIZE)) * TILE_SIZE;
    int h = tile_h + ty;
    int w = tile_w + tx;
    
    // Initialize shared memory
    if (tx < BLOCK_SIZE && ty < BLOCK_SIZE) {
        s_partial_grad_weights[ty][tx] = 0.0f;
    }
    
    // Load gradient and apply ReLU derivative
    if (h < output_height && w < output_width && b < batch_size) {
        int output_idx = ((b * out_channels + oc) * output_height + h) * output_width + w;
        const float alpha = 0.01f;  // Leaky ReLU slope
        float val = relu_output[output_idx];
        float relu_deriv = (val > 0.0f) ? 1.0f : alpha;
        s_grad[ty][tx] = grad_output[output_idx] * relu_deriv;
    } else {
        s_grad[ty][tx] = 0.0f;
    }
    
    __syncthreads();
    
    // Process each input channel
    for (int ic = 0; ic < in_channels; ic++) {
        // Load input tile into shared memory
        for (int i = ty; i < TILE_SIZE + kernel_size - 1; i += BLOCK_SIZE) {
            for (int j = tx; j < TILE_SIZE + kernel_size - 1; j += BLOCK_SIZE) {
                int ih = tile_h * stride - padding + i;
                int iw = tile_w * stride - padding + j;
                
                if (ih >= 0 && ih < height && iw >= 0 && iw < width && b < batch_size) {
                    s_input[i][j] = input[((b * in_channels + ic) * height + ih) * width + iw];
                } else {
                    s_input[i][j] = 0.0f;
                }
            }
        }
        
        // Load weights into shared memory
        if (ty < kernel_size && tx < kernel_size) {
            s_weights[ty][tx] = weights[((oc * in_channels + ic) * kernel_size + ty) * kernel_size + tx];
        }
        
        __syncthreads();
        
        // Compute gradients
        if (h < output_height && w < output_width && b < batch_size) {
            // Compute input gradients
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ih = h * stride - padding + kh;
                    int iw = w * stride - padding + kw;
                    
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        int input_idx = ((b * in_channels + ic) * height + ih) * width + iw;
                        atomicAdd(&grad_input[input_idx], 
                                s_grad[ty][tx] * s_weights[kh][kw]);
                    }
                }
            }
            
            // Accumulate weight gradients in shared memory
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ih = h * stride - padding + kh;
                    int iw = w * stride - padding + kw;
                    
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        s_partial_grad_weights[kh][kw] += s_input[ih - tile_h + padding][iw - tile_w + padding] * s_grad[ty][tx];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Accumulate weight gradients globally
    if (ty < kernel_size && tx < kernel_size) {
        int weight_idx = ((oc * in_channels) * kernel_size + ty) * kernel_size + tx;
        atomicAdd(&grad_weights[weight_idx], s_partial_grad_weights[ty][tx]);
    }
    
    // Accumulate bias gradients
    if (tx == 0 && ty == 0) {
        float bias_grad = 0.0f;
        for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                if (tile_h + i < output_height && tile_w + j < output_width) {
                    bias_grad += s_grad[i][j];
                }
            }
        }
        atomicAdd(&grad_biases[oc], bias_grad);
    }
}

__global__ void clip_gradients_kernel(float* gradients, size_t size, float max_norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = gradients[idx];
        if (val > max_norm) {
            gradients[idx] = max_norm;
        } else if (val < -max_norm) {
            gradients[idx] = -max_norm;
        }
    }
}

__global__ void clip_values_kernel(float* values, size_t size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        values[idx] = fmaxf(fminf(values[idx], max_val), min_val);
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
      d_pool_indices(nullptr), current_batch_size(0),
      streams_initialized(false) {
    
    init_streams();

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
    if (streams_initialized) {
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream3);
    }
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
    std::cout << "\nForward pass dimensions for ConvBlock:" << std::endl;
    std::cout << "Setting input_height: " << height << std::endl;
    std::cout << "Setting input_width: " << width << std::endl;
    std::cout << "Setting batch_size: " << batch_size << std::endl;
    std::cout << "in_channels: " << in_channels << std::endl;
    std::cout << "out_channels: " << out_channels << std::endl;

    // Allocate memory for this forward pass
    allocate_memory(batch_size);

    // Cache input for backward pass
    size_t input_size = batch_size * in_channels * height * width * sizeof(float);
    CHECK_CUDA_ERROR(cudaMemcpy(d_cache, d_input, input_size, cudaMemcpyDeviceToDevice));

    // Launch convolution kernel
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(batch_size,
              out_channels,
              ((height + TILE_SIZE - 1) / TILE_SIZE) * 
              ((width + TILE_SIZE - 1) / TILE_SIZE));

    conv_forward_kernel<<<grid, block>>>(
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
    if (!streams_initialized) {
        throw std::runtime_error("Streams not initialized");
    }
    // Debug dimensions
    std::cout << "\nBackward pass dimensions for ConvBlock:" << std::endl;
    std::cout << "batch_size: " << batch_size << std::endl;
    std::cout << "in_channels: " << in_channels << std::endl;
    std::cout << "input_height: " << input_height << std::endl;
    std::cout << "input_width: " << input_width << std::endl;

    // Calculate each dimension separately for debugging
    size_t dim1 = static_cast<size_t>(batch_size);
    size_t dim2 = static_cast<size_t>(in_channels);
    size_t dim3 = static_cast<size_t>(input_height);
    size_t dim4 = static_cast<size_t>(input_width);
    
    std::cout << "Dimension products:" << std::endl;
    std::cout << "dim1 (batch_size): " << dim1 << std::endl;
    std::cout << "dim1 * dim2: " << (dim1 * dim2) << std::endl;
    std::cout << "dim1 * dim2 * dim3: " << (dim1 * dim2 * dim3) << std::endl;
    std::cout << "dim1 * dim2 * dim3 * dim4: " << (dim1 * dim2 * dim3 * dim4) << std::endl;

    // Calculate total size
    size_t total_elements = dim1 * dim2 * dim3 * dim4;
    size_t total_bytes = total_elements * sizeof(float);

    std::cout << "Total elements: " << total_elements << std::endl;
    std::cout << "Total bytes to set: " << total_bytes << std::endl;
    std::cout << "d_grad_input pointer: " << d_grad_input << std::endl;

    // Verify dimensions are valid
    if (total_elements == 0) {
        throw std::runtime_error("total_elements is 0 - dimensions not properly set");
    }
    if (input_height == 0 || input_width == 0) {
        throw std::runtime_error("input dimensions are 0");
    }

    // Check for potential overflow
    if (total_bytes / sizeof(float) != total_elements) {
        throw std::runtime_error("Size calculation overflow detected");
    }

    // Verify memory pointer
    if (d_grad_input == nullptr) {
        throw std::runtime_error("d_grad_input is null");
    }

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t total_memory = prop.totalGlobalMem;
    std::cout << "Total GPU memory: " << total_memory << " bytes" << std::endl;
    std::cout << "Requested memory: " << total_bytes << " bytes" << std::endl;

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);


    // Calculate sizes
    size_t weight_size = out_channels * in_channels * kernel_size * kernel_size;
    size_t bias_size = out_channels;
    size_t input_size = static_cast<size_t>(batch_size) * 
                       static_cast<size_t>(in_channels) * 
                       static_cast<size_t>(input_height) * 
                       static_cast<size_t>(input_width);

    // Allocate temporary gradient buffers
    float *d_grad_weights, *d_grad_biases;
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_weights, weight_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_biases, bias_size * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemset(d_grad_input, 0, input_size * sizeof(float)));
    
    // Zero out gradients
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_grad_weights, 0, weight_size * sizeof(float), stream1));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_grad_biases, 0, bias_size * sizeof(float), stream2));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_grad_input, 0, input_size * sizeof(float), stream3));

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    // Allocate and initialize unpooled gradients
    float* d_unpooled_grad;
    size_t conv_output_size = batch_size * out_channels * conv_output_height * conv_output_width;
    CHECK_CUDA_ERROR(cudaMalloc(&d_unpooled_grad, conv_output_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_unpooled_grad, 0, conv_output_size * sizeof(float), stream1));
    
    cudaStreamSynchronize(stream1);

    // Launch max pooling backward
    dim3 gridDimPool(batch_size, 
                    out_channels, 
                    (pool_output_height * pool_output_width + MAX_THREADS - 1) / MAX_THREADS);
    dim3 blockDimPool(MAX_THREADS);
    
    max_pool_backward_kernel<<<gridDimPool, blockDimPool, 0, stream1>>>(
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
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream1);
    // Launch convolution backward
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        batch_size,
        out_channels,
        ((conv_output_height + TILE_SIZE - 1) / TILE_SIZE) * 
        ((conv_output_width + TILE_SIZE - 1) / TILE_SIZE)
    );

    conv_backward_kernel<<<gridDim, blockDim, 0, stream1>>>(
        d_unpooled_grad,
        d_weights,
        d_grad_input,
        d_grad_weights,
        d_grad_biases,
        d_cache,
        d_relu_output_cache,
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
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream1);
    
    
    // Update parameters using optimizers in separate streams
    weights_optimizer.update(d_weights, d_grad_weights, stream2);
    bias_optimizer.update(d_biases, d_grad_biases, stream3);

    const float max_grad_norm = 2.0f; 
    
    dim3 clip_block(256);
    dim3 clip_grid((weight_size + clip_block.x - 1) / clip_block.x);
    
    // Clip weight gradients
    clip_gradients_kernel<<<clip_grid, clip_block>>>(
        d_grad_weights, 
        weight_size, 
        max_grad_norm
    );
    CHECK_LAST_CUDA_ERROR();
    
    // Clip bias gradients
    dim3 clip_bias_grid((out_channels + clip_block.x - 1) / clip_block.x);
    clip_gradients_kernel<<<clip_bias_grid, clip_block>>>(
        d_grad_biases, 
        out_channels, 
        max_grad_norm
    );
    CHECK_LAST_CUDA_ERROR();


    const float max_weight_val = 2.0f;
    clip_values_kernel<<<(weight_size + 255) / 256, 256>>>(
        d_weights,
        weight_size,
        -max_weight_val,
        max_weight_val
    );
    cudaDeviceSynchronize();

    
    // Synchronize all streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in backward pass: %s\n", cudaGetErrorString(err));
    }
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_grad_weights));
    CHECK_CUDA_ERROR(cudaFree(d_grad_biases));
    CHECK_CUDA_ERROR(cudaFree(d_unpooled_grad));

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

}