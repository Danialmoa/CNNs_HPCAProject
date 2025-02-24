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

void test_maxpool_backward() {
    // Test input gradients: 2x2 output gradients
    float grad_output[4] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };
    
    // Expected gradients (4x4), only positions corresponding to max values should receive gradients
    float expected_grad_input[16] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 2.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 3.0f, 0.0f, 4.0f
    };
    
    // Allocate device memory
    float *d_grad_output, *d_grad_input;
    int *d_indices;
    cudaMalloc(&d_grad_output, 4 * sizeof(float));
    cudaMalloc(&d_grad_input, 16 * sizeof(float));
    cudaMalloc(&d_indices, 4 * sizeof(int));
    
    // Initialize grad_input to zeros
    cudaMemset(d_grad_input, 0, 16 * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_grad_output, grad_output, 4 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set indices (matching the forward pass results)
    int indices[4] = {5, 7, 13, 15};  // Corresponding to the max values from forward pass
    cudaMemcpy(d_indices, indices, 4 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch backward kernel
    dim3 block(128);
    dim3 grid((4 + block.x - 1) / block.x);
    
    max_pool_backward_kernel<<<grid, block>>>(
        d_grad_output, d_grad_input, d_indices,
        1, 1, 4, 4, 2, 2, 2, 2
    );
    
    // Get results
    float grad_input[16];
    cudaMemcpy(grad_input, d_grad_input, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "\nMaxPool Backward Test Results:\n";
    std::cout << "Gradient Input (4x4):\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << grad_input[i*4 + j] << "\t";
        }
        std::cout << "\n";
    }
    
    // Cleanup
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_indices);
}

int main() {
    try {
        test_simple_maxpool();
        test_maxpool_backward();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}