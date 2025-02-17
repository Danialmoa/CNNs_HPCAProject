#include "../include/kernels/pooling_kernels.cuh"

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

__global__ void max_pool_backward_kernel(
    const float* grad_output,
    float* grad_input,
    const int* indices,
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