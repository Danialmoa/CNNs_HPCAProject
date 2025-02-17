#include "../include/kernels/activation_kernels.cuh"
#include <iostream>

void test_simple_relu() {
    // Test input
    float input[8] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, -1.5f};
    float expected[8] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.5f, 0.0f};
    
    // Allocate device memory
    float *d_data;
    cudaMalloc(&d_data, 8 * sizeof(float));
    cudaMemcpy(d_data, input, 8 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    relu_kernel<<<1, 8>>>(d_data, 8);
    
    // Get results
    float output[8];
    cudaMemcpy(output, d_data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "ReLU Test Results:\n";
    std::cout << "Input\tExpected\tActual\n";
    for (int i = 0; i < 8; i++) {
        std::cout << input[i] << "\t" << expected[i] << "\t\t" 
                  << output[i] << "\n";
    }
    
    cudaFree(d_data);
}

int main() {
    try {
        test_simple_relu();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;

