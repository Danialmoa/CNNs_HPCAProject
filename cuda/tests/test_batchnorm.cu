#include "../include/kernels/batchnorm_kernels.cuh"
#include <iostream>

void test_simple_batchnorm() {
    // Test input: 2 channels, 2x2 feature maps
    float input[16] = {
        // Channel 1
        1.0f, 2.0f, 3.0f, 4.0f,    // First batch
        5.0f, 6.0f, 7.0f, 8.0f,    // Second batch
        // Channel 2
        1.0f, 2.0f, 3.0f, 4.0f,    // First batch
        5.0f, 6.0f, 7.0f, 8.0f     // Second batch
    };
    
    // Print input data for verification
    std::cout << "Input Data:\n";
    for (int b = 0; b < 2; b++) {
        std::cout << "Batch " << b + 1 << ":\n";
        for (int c = 0; c < 2; c++) {
            std::cout << "Channel " << c + 1 << ": ";
            for (int i = 0; i < 4; i++) {
                int idx = (b * 2 + c) * 4 + i;
                std::cout << input[idx] << " ";
            }
            std::cout << "\n";
        }
    }
    std::cout << "\n";

    float gamma[2] = {1.0f, 1.0f};
    float beta[2] = {0.0f, 0.0f};
    float running_mean[2] = {0.0f, 0.0f};
    float running_var[2] = {1.0f, 1.0f};
    
    // Allocate and initialize device memory
    float *d_input, *d_gamma, *d_beta, *d_running_mean, *d_running_var;
    float *d_batch_mean, *d_batch_var;
    
    cudaMalloc(&d_input, 16 * sizeof(float));
    cudaMalloc(&d_gamma, 2 * sizeof(float));
    cudaMalloc(&d_beta, 2 * sizeof(float));
    cudaMalloc(&d_running_mean, 2 * sizeof(float));
    cudaMalloc(&d_running_var, 2 * sizeof(float));
    cudaMalloc(&d_batch_mean, 2 * sizeof(float));
    cudaMalloc(&d_batch_var, 2 * sizeof(float));

    // Initialize batch statistics to zero
    cudaMemset(d_batch_mean, 0, 2 * sizeof(float));
    cudaMemset(d_batch_var, 0, 2 * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, input, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_running_mean, running_mean, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_running_var, running_var, 2 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    const int threadsPerBlock = 256;
    size_t shared_mem_size = 2 * threadsPerBlock * sizeof(float);
    
    batch_norm_forward_kernel<<<2, threadsPerBlock, shared_mem_size>>>(
        d_input, d_gamma, d_beta,
        d_running_mean, d_running_var,
        d_batch_mean, d_batch_var,
        2,  // batch_size
        2,  // channels
        2,  // height
        2,  // width
        1e-5f,  // epsilon
        0.1f,   // momentum
        true    // is_training
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    // Get results
    float output[16];
    float batch_mean[2], batch_var[2];
    cudaMemcpy(output, d_input, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(batch_mean, d_batch_mean, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(batch_var, d_batch_var, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results with expected values
    std::cout << "BatchNorm Test Results:\n";
    std::cout << "Channel 1 (Expected Mean: 2.5, Var: 1.25):\n";
    std::cout << "Actual Mean: " << batch_mean[0] << " Var: " << batch_var[0] << "\n";
    std::cout << "Channel 2 (Expected Mean: 6.5, Var: 1.25):\n";
    std::cout << "Actual Mean: " << batch_mean[1] << " Var: " << batch_var[1] << "\n\n";
    
    std::cout << "Normalized Output:\n";
    for (int b = 0; b < 2; b++) {
        std::cout << "Batch " << b + 1 << ":\n";
        for (int c = 0; c < 2; c++) {
            std::cout << "Channel " << c + 1 << ":\n";
            for (int h = 0; h < 2; h++) {
                for (int w = 0; w < 2; w++) {
                    int idx = ((b * 2 + c) * 2 + h) * 2 + w;
                    std::cout << output[idx] << "\t";
                }
                std::cout << "\n";
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