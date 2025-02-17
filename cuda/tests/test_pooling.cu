#include "../include/kernels/pooling_kernels.cuh"
#include <iostream>

void test_simple_maxpool() {
    // Test input: 4x4 feature map
    float input[16] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    
    // Expected output (2x2) with 2x2 pooling
    float expected[4] = {
        6.0f, 8.0f,
        14.0f, 16.0f
    };
    
    // Allocate device memory
    float *d_input, *d_output;
    int *d_indices;
    cudaMalloc(&d_input, 16 * sizeof(float));
    cudaMalloc(&d_output, 4 * sizeof(float));
    cudaMalloc(&d_indices, 4 * sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_input, input, 16 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 block(2, 2);
    dim3 grid(1, 1, 1);
    
    max_pool_kernel<<<grid, block>>>(
        d_input, d_output, d_indices,
        1, 1, 4, 4, 2, 2, 2, 2
    );
    
    // Get results
    float output[4];
    int indices[4];
    cudaMemcpy(output, d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(indices, d_indices, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "MaxPool Test Results:\n";
    std::cout << "Expected:\tActual:\tIndices:\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << expected[i*2 + j] << "\t\t" 
                      << output[i*2 + j] << "\t" 
                      << indices[i*2 + j] << "\n";
        }
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices);
}