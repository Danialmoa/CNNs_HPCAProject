#include "../include/kernels/batchnorm_kernels.cuh"

__global__ void batch_norm_forward_kernel(
    float* input,
    const float* gamma,
    const float* beta,
    float* running_mean,
    float* running_var,
    float* batch_mean,
    float* batch_var,
    int batch_size,
    int channels,
    int height,
    int width,
    float epsilon,
    float momentum,
    bool is_training
) {
    const int c = blockIdx.x;
    if (c >= channels) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int spatial_size = height * width;
    const int N = batch_size * spatial_size;

    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_sq_sum = &shared_mem[block_size];

    // Initialize accumulators
    float sum = 0.0f;
    float sq_sum = 0.0f;

    // First pass: compute sums
    for (int i = tid; i < N; i += block_size) {
        const int b = i / spatial_size;
        const int spatial_idx = i % spatial_size;
        const int h = spatial_idx / width;
        const int w = spatial_idx % width;
        const int idx = ((b * channels + c) * height + h) * width + w;
        float val = input[idx];
        sum += val;
        sq_sum += val * val;
    }

    // Store in shared memory
    shared_sum[tid] = sum;
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = block_size/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }

    // Only thread 0 updates the batch statistics
    if (tid == 0) {
        float mean = shared_sum[0] / N;
        float mean_sq = shared_sq_sum[0] / N;
        float var = mean_sq - (mean * mean);
        
        batch_mean[c] = mean;
        batch_var[c] = var;
        
        if (is_training) {
            running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * mean;
            running_var[c] = momentum * running_var[c] + (1.0f - momentum) * var;
        }
    }
    __syncthreads();

    // Second pass: normalize
    float mean = batch_mean[c];
    float var = batch_var[c];
    float inv_std = rsqrtf(var + epsilon);

    for (int i = tid; i < N; i += block_size) {
        const int b = i / spatial_size;
        const int spatial_idx = i % spatial_size;
        const int h = spatial_idx / width;
        const int w = spatial_idx % width;
        const int idx = ((b * channels + c) * height + h) * width + w;
        float normalized = (input[idx] - mean) * inv_std;
        input[idx] = gamma[c] * normalized + beta[c];
    }
}

__global__ void batch_norm_backward_kernel(
    const float* grad_output,
    const float* input,
    const float* gamma,
    const float* batch_mean,
    const float* batch_var,
    float* grad_input,
    float* grad_gamma,
    float* grad_beta,
    int batch_size,
    int channels,
    int height,
    int width,
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