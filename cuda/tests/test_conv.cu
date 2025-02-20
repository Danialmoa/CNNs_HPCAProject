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
    const float expected_output[16] = {
        -4, -1, 1, 4,
        -7, -3, 3, 7,
        -7, -3, 3, 7,
        -4, -1, 1, 4
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
    float output[16];
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
    float h_kernel[9];
    cudaMemcpy(h_kernel, d_kernel, kernel_size_bytes, cudaMemcpyDeviceToHost);
  
    for (int h = 0; h < kernel_size; h++) {
        for (int w = 0; w < kernel_size; w++) {
            std::cout << h_kernel[h * kernel_size + w] << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nConvolution Results:\n";
    std::cout << "Actual:\n";
    
    for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
            std::cout << output[i * out_width + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Expected:\n";
    for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
            std::cout << expected_output[i * out_width + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

void test_simple_convolution_backward() {
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
    const int out_height = 4;
    const int out_width = 4;
    
    std::cout << "\n=== Testing Backward Convolution ===\n";
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
    
    // Gradient with respect to output: [batch_size, out_channels, out_height, out_width]
    const float grad_output[16] = {
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1
    };
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_grad_output, *d_grad_input, *d_grad_kernel;
    const size_t input_size = batch_size * in_channels * height * width * sizeof(float);
    const size_t kernel_size_bytes = out_channels * in_channels * kernel_size * kernel_size * sizeof(float);
    const size_t output_size = batch_size * out_channels * out_height * out_width * sizeof(float);
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_kernel, kernel_size_bytes);
    cudaMalloc(&d_grad_output, output_size);
    cudaMalloc(&d_grad_input, input_size);
    cudaMalloc(&d_grad_kernel, kernel_size_bytes);
    
    // Initialize gradients to zero
    cudaMemset(d_grad_input, 0, input_size);
    cudaMemset(d_grad_kernel, 0, kernel_size_bytes);
    
    // Copy data to device
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, grad_output, output_size, cudaMemcpyHostToDevice);
    
    // Launch backward kernels
    dim3 block(8, 8);
    dim3 grid_input(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        batch_size * in_channels
    );
    
    dim3 grid_kernel(
        (kernel_size + block.x - 1) / block.x,
        (kernel_size + block.y - 1) / block.y,
        out_channels * in_channels
    );
    
    // Compute gradients
    conv_backward_kernel<<<grid_input, block>>>(
        d_grad_output,
        d_kernel,
        d_grad_input,
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
    
    conv_backward_kernel<<<grid_kernel, block>>>(
        d_grad_output,
        d_input,
        d_grad_kernel,
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
    float grad_input[16];
    float grad_kernel[9];
    cudaMemcpy(grad_input, d_grad_input, input_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_kernel, d_grad_kernel, kernel_size_bytes, cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "\nGradient w.r.t Input:\n";
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            std::cout << grad_input[h * width + w] << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nGradient w.r.t Kernel:\n";
    for (int h = 0; h < kernel_size; h++) {
        for (int w = 0; w < kernel_size; w++) {
            std::cout << grad_kernel[h * kernel_size + w] << " ";
        }
        std::cout << "\n";
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_grad_kernel);
}

int main() {
    try {
        test_simple_convolution();
        test_simple_convolution_backward();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}