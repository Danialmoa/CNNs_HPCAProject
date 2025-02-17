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
    
    // Calculate output dimensions
    const int height = 4;
    const int width = 4;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 1;
    const int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, 16 * sizeof(float));
    cudaMalloc(&d_kernel, 9 * sizeof(float));
    cudaMalloc(&d_output, 4 * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, input, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel with appropriate block size for small output
    dim3 block(2, 2);  // Since output is 2x2
    dim3 grid(1, 1);   // Single grid block is sufficient
    
    conv_forward_kernel<<<grid, block>>>(
        d_input,         // input
        d_kernel,        // weights
        nullptr,         // no bias
        d_output,        // output
        1,              // batch_size
        1,              // in_channels
        1,              // out_channels
        height,         // height
        width,          // width
        kernel_size,    // kernel_size
        stride,         // stride
        padding,        // padding
        out_height,     // out_height
        out_width       // out_width
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    // Get results
    float output[4];
    cudaMemcpy(output, d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "Convolution Test Results:\n";
    std::cout << "Expected:\tActual:\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << expected_output[i*2 + j] << "\t\t" 
                      << output[i*2 + j] << "\n";
        }
    }
    
    // Print input and kernel for verification
    std::cout << "\nInput Matrix:\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << input[i*4 + j] << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nKernel Matrix:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << kernel[i*3 + j] << " ";
        }
        std::cout << "\n";
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