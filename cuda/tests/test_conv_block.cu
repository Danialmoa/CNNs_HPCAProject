#include "../include/conv_block.cuh"

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

void print_tensor(const float* data, int batch, int channels, int height, int width, const std::string& name) {
    std::cout << "\n" << name << " [" << batch << "," << channels << "," << height << "," << width << "]:" << std::endl;
    for (int b = 0; b < std::min(batch, 2); b++) {
        std::cout << "Batch " << b << ":" << std::endl;
        for (int c = 0; c < std::min(channels, 2); c++) {
            std::cout << "Channel " << c << ":" << std::endl;
            for (int h = 0; h < std::min(height, 4); h++) {
                for (int w = 0; w < std::min(width, 4); w++) {
                    int idx = ((b * channels + c) * height + h) * width + w;
                    std::cout << std::fixed << std::setprecision(3) << data[idx] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

void test_conv_block() {
    std::cout << "\n=== Testing ConvBlock ===" << std::endl;

    // Test parameters
    const int batch_size = 2;
    const int in_channels = 3;
    const int height = 32;
    const int width = 32;
    const int out_channels = 16;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 1;
    const int pool_size = 2;
    const int pool_stride = 2;
    const float learning_rate = 0.001f;

    // Calculate output dimensions
    const int conv_out_height = (height + 2 * padding - kernel_size) / stride + 1;
    const int conv_out_width = (width + 2 * padding - kernel_size) / stride + 1;
    const int pool_out_height = (conv_out_height - pool_size) / pool_stride + 1;
    const int pool_out_width = (conv_out_width - pool_size) / pool_stride + 1;

    // Create input data
    std::vector<float> h_input(batch_size * in_channels * height * width);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    
    for (float& val : h_input) {
        val = dist(gen);
    }

    // Allocate device memory
    float *d_input, *d_output, *d_grad_output, *d_grad_input;
    size_t input_size = batch_size * in_channels * height * width * sizeof(float);
    size_t output_size = batch_size * out_channels * pool_out_height * pool_out_width * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_output, output_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_input, input_size));

    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));

    try {
        // Create ConvBlock
        std::cout << "Creating ConvBlock..." << std::endl;
        ConvBlock conv_block(in_channels, out_channels, kernel_size, 
                           stride, padding, pool_size, pool_stride, 
                           learning_rate);

        // Test forward pass
        std::cout << "Testing forward pass..." << std::endl;
        conv_block.forward(d_input, d_output, batch_size, height, width);

        // Get forward output
        std::vector<float> h_output(batch_size * out_channels * pool_out_height * pool_out_width);
        CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, output_size, cudaMemcpyDeviceToHost));

        // Print sample of input and output
        print_tensor(h_input.data(), batch_size, in_channels, height, width, "Input");
        print_tensor(h_output.data(), batch_size, out_channels, pool_out_height, pool_out_width, "Forward Output");

        // Create dummy gradients for backward pass
        std::vector<float> h_grad_output(batch_size * out_channels * pool_out_height * pool_out_width);
        for (float& val : h_grad_output) {
            val = dist(gen);
        }
        CHECK_CUDA_ERROR(cudaMemcpy(d_grad_output, h_grad_output.data(), output_size, cudaMemcpyHostToDevice));

        // Test backward pass
        std::cout << "\nTesting backward pass..." << std::endl;
        conv_block.backward(d_grad_output, d_grad_input, batch_size, height, width);

        // Get backward output
        std::vector<float> h_grad_input(batch_size * in_channels * height * width);
        CHECK_CUDA_ERROR(cudaMemcpy(h_grad_input.data(), d_grad_input, input_size, cudaMemcpyDeviceToHost));

        // Print sample of gradients
        print_tensor(h_grad_input.data(), batch_size, in_channels, height, width, "Gradient Output");

        // Verify shapes
        std::cout << "\nShape verification:" << std::endl;
        std::cout << "Input shape: [" << batch_size << "," << in_channels << "," 
                  << height << "," << width << "]" << std::endl;
        std::cout << "Output shape: [" << batch_size << "," << out_channels << "," 
                  << pool_out_height << "," << pool_out_width << "]" << std::endl;

        // Test multiple forward-backward passes
        std::cout << "\nTesting multiple iterations..." << std::endl;
        for (int i = 0; i < 3; i++) {
            conv_block.forward(d_input, d_output, batch_size, height, width);
            conv_block.backward(d_grad_output, d_grad_input, batch_size, height, width);
            std::cout << "Iteration " << i + 1 << " completed" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        throw;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);

    std::cout << "\nTest completed successfully!" << std::endl;
}

int main() {
    try {
        // Set device
        CHECK_CUDA_ERROR(cudaSetDevice(0));
        
        // Print device info
        cudaDeviceProp prop;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Testing on GPU: " << prop.name << std::endl;
        
        // Run test
        test_conv_block();
        
        // Reset device
        CHECK_CUDA_ERROR(cudaDeviceReset());
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 