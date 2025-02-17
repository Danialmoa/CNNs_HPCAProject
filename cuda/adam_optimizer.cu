#include "include/adam_optimizer.cuh"

__global__ void adam_update_kernel(
    float* params,
    float* m,
    float* v,
    const float* gradients,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float beta1_t,
    float beta2_t,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Update biased first moment estimate
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * gradients[idx];
    
    // Update biased second moment estimate
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * gradients[idx] * gradients[idx];
    
    // Compute bias-corrected first moment estimate
    float m_hat = m[idx] / (1.0f - beta1_t);
    
    // Compute bias-corrected second moment estimate
    float v_hat = v[idx] / (1.0f - beta2_t);
    
    // Update parameters
    params[idx] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
}

AdamOptimizer::AdamOptimizer(int param_size, float lr, float b1, float b2, float eps)
    : size(param_size), learning_rate(lr), beta1(b1), beta2(b2), 
      epsilon(eps), t(0) {
    if (size > 0) {
        cudaMalloc(&d_m, size * sizeof(float));
        cudaMalloc(&d_v, size * sizeof(float));
        cudaMemset(d_m, 0, size * sizeof(float));
        cudaMemset(d_v, 0, size * sizeof(float));
    }
}

AdamOptimizer::~AdamOptimizer() {
    if (size > 0) {
        cudaFree(d_m);
        cudaFree(d_v);
    }
}

void AdamOptimizer::update(float* params, const float* gradients) {
    if (size == 0) return;
    
    t++;
    float beta1_t = pow(beta1, t);
    float beta2_t = pow(beta2, t);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    adam_update_kernel<<<num_blocks, block_size>>>(
        params, d_m, d_v, gradients,
        learning_rate, beta1, beta2, epsilon,
        beta1_t, beta2_t, size
    );
}