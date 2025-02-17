#include "../include/kernels/conv_kernels.cuh"
#include <iostream>

void test_simple_convolution() {
    // Simple 4x4 input with 1 channel
    float input[16] = {
        1, 1, 1, 1,
        1, 2, 2, 1,
        1, 2, 2, 1,
        1, 1, 1, 1
    };
    
    // 3x3 kernel
    float kernel[9] = {
        1, 0, -1,
        2, 0, -2,
        1, 0, -1
    };
    
    // Expected output (2x2)
    float expected_output[4] = {
        8, -8,
        8, -8
    };
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, 16 * sizeof(float));
    cudaMalloc(&d_kernel, 9 * sizeof(float));
    cudaMalloc(&d_output, 4 * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, input, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 block(2, 2);
    dim3 grid(1, 1, 1);
    
    conv_forward_kernel<<<grid, block>>>(
        d_input, d_kernel, nullptr,  // no bias
        d_output, 1, 1, 1, 4, 4, 3, 1, 0, 2, 2
    );
    
    // Get results
    float output[4];
    cudaMemcpy(output, d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "Convolution Test Results:\n";
    std::cout << "Expected:\t Actual:\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << expected_output[i*2 + j] << "\t\t" 
                      << output[i*2 + j] << "\n";
        }
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

int main() {
    try {
        test_simple_convolution();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}