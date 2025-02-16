#include "include/conv_block.cuh"
#include <cudnn.h>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>

class CudnnHandle {
public:
    CudnnHandle() {
        CHECK_CUDNN(cudnnCreate(&handle));
    }
    ~CudnnHandle() {
        cudnnDestroy(handle);
    }
    cudnnHandle_t get() { return handle; }
private:
    cudnnHandle_t handle;
};

#define CHECK_CUDNN(expression) \
    { \
        cudnnStatus_t status = (expression); \
        if (status != CUDNN_STATUS_SUCCESS) { \
            std::cerr << "CUDNN error on line " << __LINE__ << ": " \
                      << cudnnGetErrorString(status) << std::endl; \
            throw std::runtime_error("CUDNN error"); \
        } \
    }

#define BLOCK_SIZE 16
#define KERNEL_SIZE 3
#define SHARED_MEM_SIZE ((BLOCK_SIZE + KERNEL_SIZE - 1) * (BLOCK_SIZE + KERNEL_SIZE - 1))


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
    // Shared memory for input tile and weights
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;
    float* s_weights = &shared_mem[SHARED_MEM_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    const int out_ch = blockIdx.z;
    
    if (out_ch >= out_channels) return;
    
    // Process all batches in this thread
    for (int batch = 0; batch < batch_size; batch++) {
        float sum = biases[out_ch];
        
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            // Load weights into shared memory
            if (tx < kernel_size && ty < kernel_size) {
                int weight_idx = ((out_ch * in_channels + in_ch) * kernel_size + ty) * kernel_size + tx;
                s_weights[ty * kernel_size + tx] = weights[weight_idx];
            }
            __syncthreads();
            
            // Load input tile into shared memory
            int tile_size = BLOCK_SIZE + kernel_size - 1;
            for (int i = ty; i < tile_size; i += blockDim.y) {
                for (int j = tx; j < tile_size; j += blockDim.x) {
                    int h_in = blockIdx.y * BLOCK_SIZE + i - padding;
                    int w_in = blockIdx.x * BLOCK_SIZE + j - padding;
                    
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int input_idx = ((batch * in_channels + in_ch) * height + h_in) * width + w_in;
                        s_input[i * tile_size + j] = input[input_idx];
                    } else {
                        s_input[i * tile_size + j] = 0.0f;
                    }
                }
            }
            __syncthreads();
            
            // Compute convolution using shared memory
            if (x < out_width && y < out_height) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int h_offset = ty * stride + kh;
                        int w_offset = tx * stride + kw;
                        sum += s_input[h_offset * tile_size + w_offset] * 
                               s_weights[kh * kernel_size + kw];
                    }
                }
            }
            __syncthreads();
        }
        
        // Write output
        if (x < out_width && y < out_height) {
            int out_idx = ((batch * out_channels + out_ch) * out_height + y) * out_width + x;
            output[out_idx] = sum;
        }
    }
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
      learning_rate(lr), current_batch_size(0) {
    
    // Create cuDNN handles and descriptors
    cudnnCreate(&cudnn_handle);
    
    // Create tensor descriptors
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv_descriptor));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    
    // Initialize weights with Xavier initialization
    size_t weight_size = out_channels * in_channels * kernel_size * kernel_size;
    std::vector<float> h_weights(weight_size);
    std::vector<float> h_biases(out_channels, 0.0f);
    
    float std_dev = sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (float& w : h_weights) {
        w = dist(gen);
    }
    
    // Allocate and copy weights and biases to GPU
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights, weight_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_biases, out_channels * sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, h_weights.data(), 
                               weight_size * sizeof(float), 
                               cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_biases, h_biases.data(), 
                               out_channels * sizeof(float), 
                               cudaMemcpyHostToDevice));
                               
    // Set pooling descriptor
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(
        pooling_descriptor,
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        pool_size, pool_size,    // window height and width
        0, 0,                    // vertical and horizontal padding
        pool_stride, pool_stride // vertical and horizontal stride
    ));
}

// Forward pass
void ConvBlock::forward(const float* d_input, float* d_output, 
                       int batch_size, int height, int width) {
    if (batch_size != current_batch_size) {
        allocate_memory(batch_size);
        current_batch_size = batch_size;
        
        // Set tensor descriptors
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(
            input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            batch_size, in_channels, height, width
        ));
        
        // Set filter descriptor
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(
            filter_descriptor,
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW,
            out_channels, in_channels, kernel_size, kernel_size
        ));
        
        // Set convolution descriptor
        CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
            convolution_descriptor,
            padding, padding,     // zero-padding
            stride, stride,       // stride
            1, 1,                // dilation
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT
        ));
        
        // Get output dimensions
        CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
            convolution_descriptor,
            input_descriptor,
            filter_descriptor,
            &batch_size,
            &out_channels,
            &conv_output_height,
            &conv_output_width
        ));
        
        // Set output descriptor
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(
            output_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            batch_size, out_channels, conv_output_height, conv_output_width
        ));
        
        // Find best convolution algorithm
        CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(
            cudnn_handle,
            input_descriptor,
            filter_descriptor,
            convolution_descriptor,
            output_descriptor,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &conv_algorithm
        ));
        
        // Get workspace size
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn_handle,
            input_descriptor,
            filter_descriptor,
            convolution_descriptor,
            output_descriptor,
            conv_algorithm,
            &workspace_size
        ));
        
        // Allocate workspace
        if (workspace_size > 0) {
            if (d_workspace) cudaFree(d_workspace);
            CHECK_CUDA_ERROR(cudaMalloc(&d_workspace, workspace_size));
        }
    }
    
    // Cache input for backward pass
    CHECK_CUDA_ERROR(cudaMemcpy(d_cache, d_input, 
        batch_size * in_channels * height * width * sizeof(float), 
        cudaMemcpyDeviceToDevice));
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Perform convolution
    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn_handle,
        &alpha,
        input_descriptor, d_input,
        filter_descriptor, d_weights,
        convolution_descriptor,
        conv_algorithm,
        d_workspace,
        workspace_size,
        &beta,
        output_descriptor, d_conv_output_cache
    ));
    
    // Add biases
    CHECK_CUDNN(cudnnAddTensor(
        cudnn_handle,
        &alpha,
        bias_descriptor, d_biases,
        &alpha,
        output_descriptor, d_conv_output_cache
    ));
    
    // ReLU activation
    CHECK_CUDNN(cudnnActivationForward(
        cudnn_handle,
        activation_descriptor,
        &alpha,
        output_descriptor, d_conv_output_cache,
        &beta,
        output_descriptor, d_conv_output_cache
    ));
    
    // Max pooling
    CHECK_CUDNN(cudnnPoolingForward(
        cudnn_handle,
        pooling_descriptor,
        &alpha,
        output_descriptor, d_conv_output_cache,
        &beta,
        pooling_descriptor, d_output
    ));
}

// Backward pass implementation
void ConvBlock::backward(const float* d_grad_output, float* d_grad_input, int batch_size) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Pooling backward
    CHECK_CUDNN(cudnnPoolingBackward(
        cudnn_handle,
        pooling_descriptor,
        &alpha,
        pooling_descriptor, d_output,
        pooling_descriptor, d_grad_output,
        output_descriptor, d_conv_output_cache,
        &beta,
        output_descriptor, d_conv_grad_cache
    ));
    
    // ReLU backward
    CHECK_CUDNN(cudnnActivationBackward(
        cudnn_handle,
        activation_descriptor,
        &alpha,
        output_descriptor, d_conv_output_cache,
        output_descriptor, d_conv_grad_cache,
        output_descriptor, d_conv_output_cache,
        &beta,
        output_descriptor, d_conv_grad_cache
    ));
    
    // Convolution backward data
    CHECK_CUDNN(cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        filter_descriptor, d_weights,
        output_descriptor, d_conv_grad_cache,
        convolution_descriptor,
        conv_bwd_data_algo,
        d_workspace,
        workspace_size,
        &beta,
        input_descriptor, d_grad_input
    ));
    
    // Convolution backward filter
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(
        cudnn_handle,
        &alpha,
        input_descriptor, d_cache,
        output_descriptor, d_conv_grad_cache,
        convolution_descriptor,
        conv_bwd_filter_algo,
        d_workspace,
        workspace_size,
        &beta,
        filter_descriptor, d_grad_weights
    ));
    
    // Convolution backward bias
    CHECK_CUDNN(cudnnConvolutionBackwardBias(
        cudnn_handle,
        &alpha,
        output_descriptor, d_conv_grad_cache,
        &beta,
        bias_descriptor, d_grad_biases
    ));
    
    // Update weights and biases using Adam optimizer
    weights_optimizer.update(d_weights, d_grad_weights, stream1);
    bias_optimizer.update(d_biases, d_grad_biases, stream2);
}

// Destructor
ConvBlock::~ConvBlock() {
    // Destroy cuDNN handles and descriptors
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyTensorDescriptor(conv_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroyPoolingDescriptor(pooling_descriptor);
    cudnnDestroy(cudnn_handle);
    
    // Free GPU memory
    free_memory();
    if (d_workspace) cudaFree(d_workspace);
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