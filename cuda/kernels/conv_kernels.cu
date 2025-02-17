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
