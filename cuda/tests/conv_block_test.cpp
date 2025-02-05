#include "../include/conv_block.cuh"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

// Helper function to check if two floating point values are approximately equal
bool is_close(float a, float b, float rtol = 1e-5, float atol = 1e-8) {
    return std::abs(a - b) <= (atol + rtol * std::abs(b));
}

// Helper function to print tensor statistics
void print_tensor_stats(const std::vector<float>& tensor, const std::string& name) {
    float min_val = tensor[0], max_val = tensor[0], sum = 0;
    for (float val : tensor) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
    }
    float mean = sum / tensor.size();
    
    std::cout << name << " stats:" << std::endl;
    std::cout << "  Min: " << min_val << ", Max: " << max_val;
    std::cout << ", Mean: " << mean << ", Size: " << tensor.size() << std::endl;
}

// Test simple forward pass with known values
void test_simple_forward() {
    std::cout << "\nTesting simple forward pass..." << std::endl;
    
    // Create a small ConvBlock
    int in_channels = 2;
    int out_channels = 2;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    int pool_size = 2;
    int pool_stride = 2;
    float learning_rate = 0.01;
    
    ConvBlock conv_block(in_channels, out_channels, kernel_size, 
                        stride, padding, pool_size, pool_stride, learning_rate);
    
    // Create input tensor (batch_size=1, channels=2, height=4, width=4)
    int batch_size = 1;
    int height = 4;
    int width = 4;
    std::vector<float> h_input(batch_size * in_channels * height * width, 1.0f);
    
    // Allocate GPU memory for input and output
    float *d_input, *d_output;
    int output_height = 2;  // After pooling: (4 + 2*1 - 3)/1 + 1 = 4, then (4 - 2)/2 + 1 = 2
    int output_width = 2;   // Same as output_height
    
    cudaMalloc(&d_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * output_height * output_width * sizeof(float));
    
    // Copy input to GPU
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Forward pass
    conv_block.forward(d_input, d_output, batch_size, height, width);
    
    // Copy output back to CPU
    std::vector<float> h_output(batch_size * out_channels * output_height * output_width);
    cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print statistics
    print_tensor_stats(h_input, "Input");
    print_tensor_stats(h_output, "Output");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

// Test backward pass
void test_backward() {
    std::cout << "\nTesting backward pass..." << std::endl;
    
    // Similar setup as forward pass
    int in_channels = 2;
    int out_channels = 2;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    int pool_size = 2;
    int pool_stride = 2;
    float learning_rate = 0.01;
    
    ConvBlock conv_block(in_channels, out_channels, kernel_size, 
                        stride, padding, pool_size, pool_stride, learning_rate);
    
    int batch_size = 1;
    int height = 4;
    int width = 4;
    
    // Allocate memory for forward and backward passes
    float *d_input, *d_output, *d_grad_output, *d_grad_input;
    int output_height = 2;
    int output_width = 2;
    
    cudaMalloc(&d_input, batch_size * in_channels * height * width * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * output_height * output_width * sizeof(float));
    cudaMalloc(&d_grad_output, batch_size * out_channels * output_height * output_width * sizeof(float));
    cudaMalloc(&d_grad_input, batch_size * in_channels * height * width * sizeof(float));
    
    // Create and copy input data
    std::vector<float> h_input(batch_size * in_channels * height * width, 1.0f);
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Forward pass
    conv_block.forward(d_input, d_output, batch_size, height, width);
    
    // Create gradient and copy to GPU
    std::vector<float> h_grad_output(batch_size * out_channels * output_height * output_width, 1.0f);
    cudaMemcpy(d_grad_output, h_grad_output.data(), h_grad_output.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Backward pass
    conv_block.backward(d_grad_output, d_grad_input, batch_size);
    
    // Copy gradients back to CPU
    std::vector<float> h_grad_input(batch_size * in_channels * height * width);
    cudaMemcpy(h_grad_input.data(), d_grad_input, h_grad_input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print statistics
    print_tensor_stats(h_grad_input, "Input Gradients");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
}

// Test multiple forward and backward passes
void test_training_loop() {
    std::cout << "\nTesting training loop..." << std::endl;
    
    ConvBlock conv_block(2, 2, 3, 1, 1, 2, 2, 0.01);
    int batch_size = 2;
    int height = 4;
    int width = 4;
    int output_height = 2;
    int output_width = 2;
    
    float *d_input, *d_output, *d_grad_output, *d_grad_input;
    
    cudaMalloc(&d_input, batch_size * 2 * height * width * sizeof(float));
    cudaMalloc(&d_output, batch_size * 2 * output_height * output_width * sizeof(float));
    cudaMalloc(&d_grad_output, batch_size * 2 * output_height * output_width * sizeof(float));
    cudaMalloc(&d_grad_input, batch_size * 2 * height * width * sizeof(float));
    
    // Run multiple training iterations
    for (int i = 0; i < 5; i++) {
        std::cout << "Training iteration " << i << std::endl;
        
        // Forward pass
        conv_block.forward(d_input, d_output, batch_size, height, width);
        
        // Backward pass
        conv_block.backward(d_grad_output, d_grad_input, batch_size);
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
}

int main() {
    try {
        // Run all tests
        test_simple_forward();
        test_backward();
        test_training_loop();
        
        std::cout << "\nAll tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}