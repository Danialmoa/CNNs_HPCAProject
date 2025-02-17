#include "include/adam_optimizer.cuh"
#include <stdexcept>
#include <iostream> 

// CUDA kernel for Adam update
__global__ void adam_update_kernel(
    float* params, const float* gradients,
    float* m, float* v,
    float lr, float beta1, float beta2, float epsilon,
    float beta1_t, float beta2_t,
    size_t num_params) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_params) return;

    // Update biased first moment estimate
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * gradients[idx];
    
    // Update biased second raw moment estimate
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * gradients[idx] * gradients[idx];
    
    // Compute bias-corrected first moment estimate
    float m_hat = m[idx] / (1.0f - beta1_t);
    
    // Compute bias-corrected second raw moment estimate
    float v_hat = v[idx] / (1.0f - beta2_t);
    
    // Update parameters with PyTorch-style update
    float update = lr * m_hat / (sqrtf(v_hat) + epsilon);
    
    // Global gradient clipping (PyTorch default is typically higher)
    const float max_grad_norm = 1.0f;
    update = fmaxf(fminf(update, max_grad_norm), -max_grad_norm);
    
    params[idx] -= update;
}

AdamOptimizer::AdamOptimizer(float lr, float b1, float b2, float eps)
    : learning_rate(lr), 
      beta1(b1 == 0.0f ? 0.9f : b1),     // PyTorch default: 0.9
      beta2(b2 == 0.0f ? 0.999f : b2),   // PyTorch default: 0.999
      epsilon(eps == 0.0f ? 1e-8f : eps), // PyTorch default: 1e-8
      t(0),
      d_m(nullptr), d_v(nullptr), param_size(0) {}

AdamOptimizer::~AdamOptimizer() {
    free_memory();
}

void AdamOptimizer::allocate_memory(size_t size) {
    if (param_size != size) {
        free_memory();
        param_size = size;
        CHECK_CUDA_ERROR(cudaMalloc(&d_m, size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_v, size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_m, 0, size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_v, 0, size * sizeof(float)));
    }
}

void AdamOptimizer::free_memory() {
    if (d_m) {
        cudaFree(d_m);
        d_m = nullptr;
    }
    if (d_v) {
        cudaFree(d_v);
        d_v = nullptr;
    }
    param_size = 0;
}

void AdamOptimizer::init(size_t num_params) {
    allocate_memory(num_params);
    t = 0;
}

void AdamOptimizer::update(float* d_params, const float* d_gradients, cudaStream_t stream) {
    if (!d_m || !d_v) {
        throw std::runtime_error("Adam optimizer not initialized!");
    }

    t++;
    float beta1_t = 1.0f - std::pow(beta1, t);
    float beta2_t = 1.0f - std::pow(beta2, t);

    // Calculate number of blocks needed
    const int block_size = 256;
    const int num_blocks = (param_size + block_size - 1) / block_size;

    // Launch kernel
    adam_update_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_params, d_gradients,
        d_m, d_v,
        learning_rate, beta1, beta2, epsilon,
        beta1_t, beta2_t,
        param_size
    );
    CHECK_LAST_CUDA_ERROR();
}