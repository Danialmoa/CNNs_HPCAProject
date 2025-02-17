#include "../include/kernels/conv_kernels.cuh"

__global__ void conv_forward_kernel(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
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
    // Calculate output position
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= out_width || y >= out_height) return;
    
    // Initialize output value
    float sum = 0.0f;
    
    // Compute input window boundaries
    const int start_h = y * stride - padding;
    const int start_w = x * stride - padding;
    
    // Compute convolution
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int in_h = start_h + kh;
            int in_w = start_w + kw;
            
            if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                // Correct indexing for input and weights
                float in_val = input[in_h * width + in_w];
                float weight_val = weights[(kernel_size - 1 - kh) * kernel_size + (kernel_size - 1 - kw)];
                sum += in_val * weight_val;
            }
        }
    }
    
    // Write output
    output[y * out_width + x] = sum;
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
