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
    const int out_height = 4;  // Fixed for this test case
    const int out_width = 4;   // Fixed for this test case
    
    std::cout << "Input dimensions: " << height << "x" << width << "\n";
    std::cout << "Output dimensions: " << out_height << "x" << out_width << "\n";
    
    // Input: [batch_size, in_channels, height, width]
    const float input[16] = {
        1, 1, 1, 1,
        1, 2, 2, 1,
        1, 2, 2, 1,
        1, 1, 1, 1
    };
    
    // Kernel: [out_channels, in_channels, kernel_size, kernel_size]
    const float kernel[9] = {
        1, 0, -1,
        2, 0, -2,
        1, 0, -1
    };
    
    // Expected output: [batch_size, out_channels, out_height, out_width]
    const float expected_output[4] = {
        8, -8,
        8, -8
    };
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    const size_t input_size = batch_size * in_channels * height * width * sizeof(float);
    const size_t kernel_size_bytes = out_channels * in_channels * kernel_size * kernel_size * sizeof(float);
    const size_t output_size = batch_size * out_channels * out_height * out_width * sizeof(float);
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_kernel, kernel_size_bytes);
    cudaMalloc(&d_output, output_size);
    
    // Initialize output to zero
    cudaMemset(d_output, 0, output_size);
    
    // Copy data to device
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size_bytes, cudaMemcpyHostToDevice);
    
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
    
    // Synchronize and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Get results
    float output[4];
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "\nInput Matrix:\n";
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            std::cout << input[h * width + w] << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nKernel Matrix:\n";
    for (int h = 0; h < kernel_size; h++) {
        for (int w = 0; w < kernel_size; w++) {
            std::cout << kernel[h * kernel_size + w] << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nConvolution Results:\n";
    std::cout << "Expected:\tActual:\n";
    for (int h = 0; h < out_height; h++) {
        for (int w = 0; w < out_width; w++) {
            int idx = h * out_width + w;
            std::cout << expected_output[idx] << "\t\t" << output[idx] << "\n";
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