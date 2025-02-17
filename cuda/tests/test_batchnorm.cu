#include "../include/kernels/batchnorm_kernels.cuh"
#include <iostream>

void test_simple_batchnorm() {
    // Test input: 2 channels, 2x2 feature maps
    float input[16] = {
        // Channel 1
        1.0f, 2.0f,
        3.0f, 4.0f,
        // Channel 2
        5.0f, 6.0f,
        7.0f, 8.0f
    };
    
    float gamma[2] = {1.0f, 1.0f};
    float beta[2] = {0.0f, 0.0f};
    float running_mean[2] = {0.0f, 0.0f};
    float running_var[2] = {1.0f, 1.0f};
    
    // Allocate device memory
    float *d_input, *d_gamma, *d_beta, *d_running_mean, *d_running_var;
    float *d_batch_mean, *d_batch_var;
    
    cudaMalloc(&d_input, 16 * sizeof(float));
    cudaMalloc(&d_gamma, 2 * sizeof(float));
    cudaMalloc(&d_beta, 2 * sizeof(float));
    cudaMalloc(&d_running_mean, 2 * sizeof(float));
    cudaMalloc(&d_running_var, 2 * sizeof(float));
    cudaMalloc(&d_batch_mean, 2 * sizeof(float));
    cudaMalloc(&d_batch_var, 2 * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, input, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_running_mean, running_mean, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_running_var, running_var, 2 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    batch_norm_forward_kernel<<<2, 4>>>(
        d_input, d_gamma, d_beta,
        d_running_mean, d_running_var,
        d_batch_mean, d_batch_var,
        2, 2, 2, 2, 1e-5f, 0.1f, true
    );
    
    // Get results
    float output[16];
    float batch_mean[2], batch_var[2];
    cudaMemcpy(output, d_input, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(batch_mean, d_batch_mean, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(batch_var, d_batch_var, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "BatchNorm Test Results:\n";
    std::cout << "Channel 1 Mean: " << batch_mean[0] << " Var: " << batch_var[0] << "\n";
    std::cout << "Channel 2 Mean: " << batch_mean[1] << " Var: " << batch_var[1] << "\n";
    std::cout << "Normalized Output:\n";
    for (int c = 0; c < 2; c++) {
        std::cout << "Channel " << c+1 << ":\n";
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                std::cout << output[c*4 + i*2 + j] << "\t";
            }
            std::cout << "\n";
        }
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_running_mean);
    cudaFree(d_running_var);
    cudaFree(d_batch_mean);
    cudaFree(d_batch_var);
}

int main() {
    try {
        test_simple_batchnorm();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
