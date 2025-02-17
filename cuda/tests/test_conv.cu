#include "../include/kernels/conv_kernels.cuh"
#include <iostream>

void test_simple_convolution() {
    // Test parameters
    const int batch_size = 1;
    const int in_channels = 1;
    const int out_channels = 1;
    const int height = 4;
    const int width = 4;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 1;
    
    // Calculate output dimensions
    const int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    // Input: [batch_size, in_channels, height, width]
    float input[16] = {
        1, 1, 1, 1,
        1, 2, 2, 1,
        1, 2, 2, 1,
        1, 1, 1, 1
    };
    
    // Kernel: [out_channels, in_channels, kernel_size, kernel_size]
    float kernel[9] = {
        1, 0, -1,
        2, 0, -2,
        1, 0, -1
    };
    
    // Expected output: [batch_size, out_channels, out_height, out_width]
    float expected_output[4] = {
        8, -8,
        8, -8
    };
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, batch_size * in_channels * height * width * sizeof(float));
    cudaMalloc(&d_kernel, out_channels * in_channels * kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * out_height * out_width * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, input, batch_size * in_channels * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, out_channels * in_channels * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 block(8, 8);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        batch_size * out_channels
    );
    
    conv_forward_kernel<<<grid, block>>>(
        d_input,
        d_kernel,
        nullptr,
        d_output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        out_height,
        out_width
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Synchronize and get results
    cudaDeviceSynchronize();
    
    float output[4];
    cudaMemcpy(output, d_output, batch_size * out_channels * out_height * out_width * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "Convolution Test Results:\n";
    std::cout << "Expected:\tActual:\n";
    for (int h = 0; h < out_height; h++) {
        for (int w = 0; w < out_width; w++) {
            std::cout << expected_output[h * out_width + w] << "\t\t" 
                      << output[h * out_width + w] << "\n";
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