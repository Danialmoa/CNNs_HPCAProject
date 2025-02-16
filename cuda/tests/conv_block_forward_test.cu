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
        const int in_channels = 1;
        const int hidden_channels = 8;
        const int out_channels = 2;
        const int height = 16;
        const int width = 16;
        const int kernel_size = 3;
        const int stride = 1;
        const int padding = 1;
        const int pool_size = 2;
        const int pool_stride = 2;
        const float learning_rate = 0.01f;

        // Calculate output dimensions for conv1
        const int conv1_out_height = (height + 2 * padding - kernel_size) / stride + 1;
        const int conv1_out_width = (width + 2 * padding - kernel_size) / stride + 1;
        const int pool1_out_height = (conv1_out_height - pool_size) / pool_stride + 1;
        const int pool1_out_width = (conv1_out_width - pool_size) / pool_stride + 1;

        // Calculate output dimensions for conv2
        const int conv2_out_height = (pool1_out_height + 2 * padding - kernel_size) / stride + 1;
        const int conv2_out_width = (pool1_out_width + 2 * padding - kernel_size) / stride + 1;
        const int pool2_out_height = (conv2_out_height - pool_size) / pool_stride + 1;
        const int pool2_out_width = (conv2_out_width - pool_size) / pool_stride + 1;

        // Create ConvBlocks
        ConvBlock conv1(in_channels, hidden_channels, kernel_size, 
                       stride, padding, pool_size, pool_stride, 
                       learning_rate);
        
        ConvBlock conv2(hidden_channels, out_channels, kernel_size, 
                       stride, padding, pool_size, pool_stride, 
                       learning_rate);

        // Run two iterations
        for (int iteration = 0; iteration < 2; ++iteration) {
            std::cout << "\n=== Iteration " << iteration + 1 << " ===\n";

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

            // Allocate device memory for input, intermediate, and output
            float *d_input, *d_conv1_output, *d_final_output;
            cudaMalloc(&d_input, h_input.size() * sizeof(float));
            cudaMalloc(&d_conv1_output, batch_size * hidden_channels * pool1_out_height * pool1_out_width * sizeof(float));
            cudaMalloc(&d_final_output, batch_size * out_channels * pool2_out_height * pool2_out_width * sizeof(float));

            // Copy input to device
            cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);

            // Print configuration
            std::cout << "\n=== Test Configuration ===";
            std::cout << "\nBatch size: " << batch_size;
            std::cout << "\nInput channels: " << in_channels;
            std::cout << "\nHidden channels: " << hidden_channels;
            std::cout << "\nOutput channels: " << out_channels;
            std::cout << "\nInput size: " << height << "x" << width;
            std::cout << "\nConv1 output size: " << conv1_out_height << "x" << conv1_out_width;
            std::cout << "\nPool1 output size: " << pool1_out_height << "x" << pool1_out_width;
            std::cout << "\nConv2 output size: " << conv2_out_height << "x" << conv2_out_width;
            std::cout << "\nPool2 output size: " << pool2_out_height << "x" << pool2_out_width;
            std::cout << "\n";

            // Print first batch of input
            print_tensor(h_input.data(), in_channels, height, width, "Input (first batch)");

            // Forward pass through conv1
            conv1.forward(d_input, d_conv1_output, batch_size, height, width);

            // Get intermediate results from conv1
            std::vector<float> h_conv1_output(batch_size * hidden_channels * pool1_out_height * pool1_out_width);
            cudaMemcpy(h_conv1_output.data(), d_conv1_output, 
                      h_conv1_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
            print_tensor(h_conv1_output.data(), hidden_channels, pool1_out_height, pool1_out_width, 
                        "After Conv1 and Pool1 (first batch)");

            // Forward pass through conv2
            conv2.forward(d_conv1_output, d_final_output, batch_size, pool1_out_height, pool1_out_width);

            // Get final output
            std::vector<float> h_final_output(batch_size * out_channels * pool2_out_height * pool2_out_width);
            cudaMemcpy(h_final_output.data(), d_final_output, 
                      h_final_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
            print_tensor(h_final_output.data(), out_channels, pool2_out_height, pool2_out_width, 
                        "Final Output (first batch)");

            std::cout << "\n=== Testing Backward Pass ===\n";

            // Create gradient for backward pass
            std::vector<float> h_grad_output(batch_size * out_channels * pool2_out_height * pool2_out_width, 1.0f);

            // Allocate device memory for gradients
            float *d_grad_output, *d_grad_conv1_output, *d_grad_input;
            cudaMalloc(&d_grad_output, h_grad_output.size() * sizeof(float));
            cudaMalloc(&d_grad_conv1_output, batch_size * hidden_channels * pool1_out_height * pool1_out_width * sizeof(float));
            cudaMalloc(&d_grad_input, batch_size * in_channels * height * width * sizeof(float));

            // Copy gradient to device
            cudaMemcpy(d_grad_output, h_grad_output.data(), 
                      h_grad_output.size() * sizeof(float), 
                      cudaMemcpyHostToDevice);

            // Print gradient input
            print_tensor(h_grad_output.data(), out_channels, pool2_out_height, pool2_out_width, 
                        "Gradient Input (first batch)");

            // Backward pass
            conv2.backward(d_grad_output, d_grad_conv1_output, batch_size, pool1_out_height, pool1_out_width);
            conv1.backward(d_grad_conv1_output, d_grad_input, batch_size, height, width);

            // Get gradients with respect to input
            std::vector<float> h_grad_input(batch_size * in_channels * height * width);
            cudaMemcpy(h_grad_input.data(), d_grad_input, 
                      h_grad_input.size() * sizeof(float), 
                      cudaMemcpyDeviceToHost);

            // Print input gradients
            print_tensor(h_grad_input.data(), in_channels, height, width, 
                        "Gradient w.r.t Input (first batch)");

            // Get updated weights and biases for both layers
            std::cout << "\nConv1 Updated Parameters:";
            std::vector<float> h_conv1_weights(hidden_channels * in_channels * kernel_size * kernel_size);
            std::vector<float> h_conv1_biases(hidden_channels);
            cudaMemcpy(h_conv1_weights.data(), conv1.get_weights(), 
                      h_conv1_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_conv1_biases.data(), conv1.get_biases(), 
                      h_conv1_biases.size() * sizeof(float), cudaMemcpyDeviceToHost);

            std::cout << "\nConv1 Weights (first few values):";
            for (int i = 0; i < std::min(10, (int)h_conv1_weights.size()); i++) {
                std::cout << h_conv1_weights[i] << " ";
            }
            std::cout << "...\n";

            std::cout << "\nConv1 Biases:";
            for (int i = 0; i < hidden_channels; i++) {
                std::cout << "\nChannel " << i << ": " << h_conv1_biases[i];
            }

            std::cout << "\n\nConv2 Updated Parameters:";
            std::vector<float> h_conv2_weights(out_channels * hidden_channels * kernel_size * kernel_size);
            std::vector<float> h_conv2_biases(out_channels);
            cudaMemcpy(h_conv2_weights.data(), conv2.get_weights(), 
                      h_conv2_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_conv2_biases.data(), conv2.get_biases(), 
                      h_conv2_biases.size() * sizeof(float), cudaMemcpyDeviceToHost);

            std::cout << "\nConv2 Weights (first few values):";
            for (int i = 0; i < std::min(10, (int)h_conv2_weights.size()); i++) {
                std::cout << h_conv2_weights[i] << " ";
            }
            std::cout << "...\n";

            std::cout << "\nConv2 Biases:";
            for (int i = 0; i < out_channels; i++) {
                std::cout << "\nChannel " << i << ": " << h_conv2_biases[i];
            }

            // Cleanup iteration-specific memory
            cudaFree(d_input);
            cudaFree(d_conv1_output);
            cudaFree(d_final_output);
            cudaFree(d_grad_output);
            cudaFree(d_grad_conv1_output);
            cudaFree(d_grad_input);

            // Check for any CUDA errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
            }

            std::cout << "\n\nIteration " << iteration + 1 << " completed successfully!\n";
        }

        std::cout << "\nAll iterations completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}