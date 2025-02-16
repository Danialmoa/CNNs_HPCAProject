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

        // Create ConvBlock
        ConvBlock conv_block(in_channels, out_channels, kernel_size, 
                           stride, padding, pool_size, pool_stride, 
                           learning_rate);

        // Allocate device memory for input and output
        float *d_input, *d_output;
        cudaMalloc(&d_input, h_input.size() * sizeof(float));
        cudaMalloc(&d_output, batch_size * out_channels * pool_out_height * pool_out_width * sizeof(float));

        // Copy input to device
        cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Print configuration
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

        // Get intermediate results from conv_block's cache
        std::vector<float> h_conv_output(batch_size * out_channels * conv_out_height * conv_out_width);
        cudaMemcpy(h_conv_output.data(), conv_block.get_conv_output_cache(), 
                  h_conv_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
        print_tensor(h_conv_output.data(), out_channels, conv_out_height, conv_out_width, 
                    "After Convolution and ReLU (first batch)");

        // Get final output
        std::vector<float> h_output(batch_size * out_channels * pool_out_height * pool_out_width);
        cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
        print_tensor(h_output.data(), out_channels, pool_out_height, pool_out_width, 
                    "Final Output (first batch)");

        // After forward pass, create a simple gradient for backward pass
        std::vector<float> h_grad_output(batch_size * out_channels * pool_out_height * pool_out_width);
        
        // Initialize gradient with a simple pattern (all 1.0s for simplicity)
        for (size_t i = 0; i < h_grad_output.size(); i++) {
            h_grad_output[i] = 1.0f;
        }

        // Allocate device memory for gradients
        float *d_grad_output, *d_grad_input;
        cudaMalloc(&d_grad_output, h_grad_output.size() * sizeof(float));
        cudaMalloc(&d_grad_input, batch_size * in_channels * height * width * sizeof(float));

        // Copy gradient to device
        cudaMemcpy(d_grad_output, h_grad_output.data(), 
                  h_grad_output.size() * sizeof(float), 
                  cudaMemcpyHostToDevice);

        std::cout << "\n=== Testing Backward Pass ===\n";
        
        // Print gradient input
        print_tensor(h_grad_output.data(), out_channels, pool_out_height, pool_out_width, 
                    "Gradient Input (first batch)");

        // Backward pass
        conv_block.backward(d_grad_output, d_grad_input, batch_size);

        // Get gradients with respect to input
        std::vector<float> h_grad_input(batch_size * in_channels * height * width);
        cudaMemcpy(h_grad_input.data(), d_grad_input, 
                  h_grad_input.size() * sizeof(float), 
                  cudaMemcpyDeviceToHost);

        // Print input gradients
        print_tensor(h_grad_input.data(), in_channels, height, width, 
                    "Gradient w.r.t Input (first batch)");

        // Get updated weights and biases
        std::vector<float> h_updated_weights(out_channels * in_channels * kernel_size * kernel_size);
        std::vector<float> h_updated_biases(out_channels);

        cudaMemcpy(h_updated_weights.data(), conv_block.get_weights(), 
                  h_updated_weights.size() * sizeof(float), 
                  cudaMemcpyDeviceToHost);
        cudaMemcpy(h_updated_biases.data(), conv_block.get_biases(), 
                  h_updated_biases.size() * sizeof(float), 
                  cudaMemcpyDeviceToHost);

        // Print weight updates (just a sample)
        std::cout << "\nUpdated Weights (first few values):\n";
        for (int i = 0; i < std::min(10, (int)h_updated_weights.size()); i++) {
            std::cout << h_updated_weights[i] << " ";
        }
        std::cout << "...\n";

        // Print bias updates
        std::cout << "\nUpdated Biases:\n";
        for (int i = 0; i < out_channels; i++) {
            std::cout << "Channel " << i << ": " << h_updated_biases[i] << "\n";
        }

        // Check for any CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
        }

        // Cleanup
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_grad_output);
        cudaFree(d_grad_input);

        std::cout << "\nBackward pass test completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}