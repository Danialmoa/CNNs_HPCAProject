#include "../include/conv_block.cuh"
#include <iostream>
#include <vector>
#include <iomanip>

// Helper function to print 3D tensor (channel, height, width)
void print_tensor(const float* data, int channels, int height, int width, const std::string& name) {
    std::cout << "\n" << name << " [" << channels << ", " << height << ", " << width << "]:\n";
    for (int c = 0; c < channels; c++) {
        std::cout << "Channel " << c << ":\n";
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = (c * height + h) * width + w;
                std::cout << std::setw(6) << std::fixed << std::setprecision(2) 
                         << data[idx] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

int main() {
    try {
        // Test parameters
        const int batch_size = 2;
        const int in_channels = 3;
        const int out_channels = 4;
        const int height = 8;
        const int width = 8;
        const int kernel_size = 3;
        const int stride = 1;
        const int padding = 1;
        const int pool_size = 2;
        const int pool_stride = 2;
        const float learning_rate = 0.01f;

        // Create ConvBlock
        ConvBlock conv_block(in_channels, out_channels, kernel_size, 
                           stride, padding, pool_size, pool_stride, 
                           learning_rate);

        // Calculate output dimensions
        const int conv_out_height = (height + 2 * padding - kernel_size) / stride + 1;
        const int conv_out_width = (width + 2 * padding - kernel_size) / stride + 1;
        const int pool_out_height = (conv_out_height - pool_size) / pool_stride + 1;
        const int pool_out_width = (conv_out_width - pool_size) / pool_stride + 1;

        // Create input data with a recognizable pattern
        std::vector<float> h_input(batch_size * in_channels * height * width);
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < in_channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int idx = ((b * in_channels + c) * height + h) * width + w;
                        // Create a checkerboard pattern
                        h_input[idx] = ((h + w) % 2) * (c + 1);
                    }
                }
            }
        }

        // Allocate device memory for input and output
        float *d_input, *d_output;
        cudaMalloc(&d_input, h_input.size() * sizeof(float));
        cudaMalloc(&d_output, batch_size * out_channels * pool_out_height * pool_out_width * sizeof(float));

        // Copy input to device
        cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Print input tensor
        std::cout << "\n=== Test Configuration ===";
        std::cout << "\nBatch size: " << batch_size;
        std::cout << "\nInput channels: " << in_channels;
        std::cout << "\nOutput channels: " << out_channels;
        std::cout << "\nInput size: " << height << "x" << width;
        std::cout << "\nKernel size: " << kernel_size;
        std::cout << "\nStride: " << stride;
        std::cout << "\nPadding: " << padding;
        std::cout << "\nPool size: " << pool_size;
        std::cout << "\nPool stride: " << pool_stride;
        std::cout << "\nConv output size: " << conv_out_height << "x" << conv_out_width;
        std::cout << "\nPool output size: " << pool_out_height << "x" << pool_out_width;
        std::cout << "\n";

        // Print first batch of input
        print_tensor(h_input.data(), in_channels, height, width, "Input (first batch)");

        // Forward pass
        conv_block.forward(d_input, d_output, batch_size, height, width);

        // Get output
        std::vector<float> h_output(batch_size * out_channels * pool_out_height * pool_out_width);
        cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // Print first batch of output
        print_tensor(h_output.data(), out_channels, pool_out_height, pool_out_width, 
                    "Output (first batch)");

        // Check for any CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
        }

        // Test different input sizes
        std::vector<std::pair<int, int>> test_sizes = {
            {16, 16}, {32, 32}, {64, 64}
        };

        std::cout << "\n=== Testing Different Input Sizes ===\n";
        for (const auto& size : test_sizes) {
            int test_height = size.first;
            int test_width = size.second;
            
            std::cout << "\nTesting size: " << test_height << "x" << test_width << std::endl;
            
            // Reallocate input/output with new size
            std::vector<float> test_input(batch_size * in_channels * test_height * test_width, 1.0f);
            float* d_test_input;
            float* d_test_output;
            
            cudaMalloc(&d_test_input, test_input.size() * sizeof(float));
            cudaMemcpy(d_test_input, test_input.data(), test_input.size() * sizeof(float), 
                      cudaMemcpyHostToDevice);
            
            int test_conv_out_h = (test_height + 2 * padding - kernel_size) / stride + 1;
            int test_conv_out_w = (test_width + 2 * padding - kernel_size) / stride + 1;
            int test_pool_out_h = (test_conv_out_h - pool_size) / pool_stride + 1;
            int test_pool_out_w = (test_conv_out_w - pool_size) / pool_stride + 1;
            
            cudaMalloc(&d_test_output, batch_size * out_channels * test_pool_out_h * 
                      test_pool_out_w * sizeof(float));
            
            // Forward pass with new size
            conv_block.forward(d_test_input, d_test_output, batch_size, test_height, test_width);
            
            std::cout << "Forward pass successful for size " << test_height << "x" << test_width 
                      << std::endl;
            
            cudaFree(d_test_input);
            cudaFree(d_test_output);
        }

        // Cleanup
        cudaFree(d_input);
        cudaFree(d_output);

        std::cout << "\nAll tests completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 