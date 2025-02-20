#include "../include/kernels/conv_kernels.cuh"

__global__ void conv_forward_kernel(
    const float* input,           // [N, C_in, H, W]
    const float* weights,         // [C_out, C_in, K, K]
    const float* biases,          // [C_out]
    float* output,                // [N, C_out, H_out, W_out]
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
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z / out_channels;
    const int c_out = blockIdx.z % out_channels;
    
    if (x >= out_width || y >= out_height || n >= batch_size) return;
    
    float sum = biases ? biases[c_out] : 0.0f;
    
    // Compute the input starting position
    const int start_h = y * stride - padding;
    const int start_w = x * stride - padding;
    
    // Compute convolution
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            const int in_h = start_h + kh;
            
            for (int kw = 0; kw < kernel_size; kw++) {
                const int in_w = start_w + kw;
                
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                    const int in_idx = ((n * in_channels + c_in) * height + in_h) * width + in_w;
                    const int w_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                    sum += input[in_idx] * weights[w_idx];
                }
            }
        }
    }
    
    const int out_idx = ((n * out_channels + c_out) * out_height + y) * out_width + x;
    output[out_idx] = sum;
}

__global__ void conv_backward_kernel(
    const float* grad_output,
    const float* input,
    const float* weights,
    float* grad_input,
    float* grad_weights,
    float* grad_biases,
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
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z / out_channels;
    const int c_out = blockIdx.z % out_channels;
    
    if (x >= out_width || y >= out_height || n >= batch_size) return;
    
    // Get gradient from output
    const int out_idx = ((n * out_channels + c_out) * out_height + y) * out_width + x;
    const float grad = grad_output[out_idx];
    atomicAdd(&grad_biases[c_out], grad);

    // L2 regularization coefficient
    const float l2_reg = 0.0001f;
    
    // Compute the input window position
    const int start_h = y * stride - padding;
    const int start_w = x * stride - padding;
    
    // Compute gradients for weights and input
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            const int in_h = start_h + kh;
            
            for (int kw = 0; kw < kernel_size; kw++) {
                const int in_w = start_w + kw;
                
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                    const int in_idx = ((n * in_channels + c_in) * height + in_h) * width + in_w;
                    const int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                    
                    // Gradient w.r.t. weights
                    float weight_grad = input[in_idx] * grad + l2_reg * weights[weight_idx];
                    atomicAdd(&grad_weights[weight_idx], weight_grad);
                    
                    // Gradient w.r.t. input
                    atomicAdd(&grad_input[in_idx], weights[weight_idx] * grad);
                }
            }
        }
    }
}