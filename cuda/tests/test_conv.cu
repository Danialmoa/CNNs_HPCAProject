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
    
    // Calculate shared memory size
    const int BLOCK_SIZE = 16;
    size_t shared_mem_size = 
        ((BLOCK_SIZE + 3 - 1) * (BLOCK_SIZE + 3 - 1) + 3 * 3) * sizeof(float);
    
    // Launch kernel
    dim3 block(16, 16);  // Use a 16x16 block
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y
    );
    
    conv_forward_kernel<<<grid, block>>>(
        d_input,         // input
        d_kernel,        // weights
        nullptr,         // no bias
        d_output,        // output
        1,              // batch_size
        1,              // in_channels
        1,              // out_channels
        4,              // height
        4,              // width
        3,              // kernel_size
        1,              // stride
        1,              // padding
        2,              // out_height
        2               // out_width
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