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
        
        // Clip input values for stability
        val = max(min(val, 10.0f), -10.0f);
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
        float var = max(mean_sq - (mean * mean), epsilon);
        
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
    float var = max(batch_var[c], epsilon);
    float inv_std = rsqrtf(var);
    float gamma_val = gamma[c];
    float beta_val = beta[c];

    for (int i = tid; i < N; i += block_size) {
        const int b = i / spatial_size;
        const int spatial_idx = i % spatial_size;
        const int h = spatial_idx / width;
        const int w = spatial_idx % width;
        const int idx = ((b * channels + c) * height + h) * width + w;
        
        float val = input[idx];
        float normalized = (val - mean) * inv_std;
        // Clip normalized value
        normalized = max(min(normalized, 10.0f), -10.0f);
        float result = gamma_val * normalized + beta_val;
        // Clip final result
        result = max(min(result, 10.0f), -10.0f);
        input[idx] = result;
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

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int spatial_size = height * width;
    const int N = batch_size * spatial_size;

    extern __shared__ float shared_mem[];
    float* shared_grad = shared_mem;
    float* shared_grad_h = &shared_mem[block_size];

    // Initialize accumulators
    float sum_grad = 0.0f;
    float sum_grad_h = 0.0f;

    const float mean = batch_mean[c];
    const float var = max(batch_var[c], epsilon);
    const float inv_std = rsqrtf(var);

    // First pass: compute gradient sums
    for (int i = tid; i < N; i += block_size) {
        const int b = i / spatial_size;
        const int spatial_idx = i % spatial_size;
        const int h = spatial_idx / width;
        const int w = spatial_idx % width;
        const int idx = ((b * channels + c) * height + h) * width + w;
        
        float dout = grad_output[idx];
        float x = input[idx];
        
        // Clip gradients for stability
        dout = max(min(dout, 1.0f), -1.0f);
        
        sum_grad += dout;
        sum_grad_h += dout * (x - mean) * inv_std;
    }

    // Store in shared memory
    shared_grad[tid] = sum_grad;
    shared_grad_h[tid] = sum_grad_h;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = block_size/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_grad[tid] += shared_grad[tid + stride];
            shared_grad_h[tid] += shared_grad_h[tid + stride];
        }
        __syncthreads();
    }

    // Only thread 0 updates the gradients for gamma and beta
    if (tid == 0) {
        atomicAdd(&grad_beta[c], shared_grad[0]);
        atomicAdd(&grad_gamma[c], shared_grad_h[0]);
    }
    __syncthreads();

    // Second pass: compute gradients for input
    const float gamma_val = gamma[c];
    const float inv_N = 1.0f / N;
    const float grad_scale = gamma_val * inv_std;

    for (int i = tid; i < N; i += block_size) {
        const int b = i / spatial_size;
        const int spatial_idx = i % spatial_size;
        const int h = spatial_idx / width;
        const int w = spatial_idx % width;
        const int idx = ((b * channels + c) * height + h) * width + w;
        
        float dout = grad_output[idx];
        float x = input[idx];
        
        // Clip gradients
        dout = max(min(dout, 1.0f), -1.0f);
        
        float dx = grad_scale * (
            dout - (shared_grad[0] * inv_N) - 
            ((x - mean) * inv_std * shared_grad_h[0] * inv_N)
        );
        
        // Clip final gradient
        dx = max(min(dx, 1.0f), -1.0f);
        grad_input[idx] = dx;
    }
}