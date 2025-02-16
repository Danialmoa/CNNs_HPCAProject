#include "../include/conv_block.cuh"
#include <iostream>
#include <vector>
#include <iomanip>
#include <numeric>

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

// Simple MSE loss calculation
float calculate_loss(const float* output, const float* target, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = output[i] - target[i];
        loss += diff * diff;
    }
    return loss / size;
}

int main() {
    try {
        // Test parameters
        const int batch_size = 2;
        const int in_channels = 3;      // Input channels (e.g., RGB)
        const int hidden_channels = 8;   // Intermediate feature maps
        const int out_channels = 4;      // Output feature maps
        const int height = 32;
        const int width = 32;
        const int kernel_size = 3;
        const int stride = 1;
        const int padding = 1;
        const int pool_size = 2;
        const int pool_stride = 2;
        const float learning_rate = 0.01f;
        const int num_epochs = 2;

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

        std::cout << "\nNetwork Architecture:";
        std::cout << "\nInput: " << batch_size << "x" << in_channels << "x" << height << "x" << width;
        std::cout << "\nConv1: " << hidden_channels << " channels, " << pool1_out_height << "x" << pool1_out_width;
        std::cout << "\nConv2: " << out_channels << " channels, " << pool2_out_height << "x" << pool2_out_width;
        std::cout << "\n";

        // Training loop
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::cout << "\n=== Epoch " << epoch + 1 << " ===\n";

            // Create input data with a recognizable pattern
            std::vector<float> h_input(batch_size * in_channels * height * width);
            for (int b = 0; b < batch_size; b++) {
                for (int c = 0; c < in_channels; c++) {
                    for (int h = 0; h < height; h++) {
                        for (int w = 0; w < width; w++) {
                            int idx = ((b * in_channels + c) * height + h) * width + w;
                            // Create a checkerboard pattern
                            h_input[idx] = ((h + w) % 2) * (c + 1) + epoch * 0.1f;
                        }
                    }
                }
            }

            // Create target data (simple target for demonstration)
            std::vector<float> h_target(batch_size * out_channels * pool2_out_height * pool2_out_width, 1.0f);

            // Allocate device memory
            float *d_input, *d_conv1_output, *d_final_output;
            cudaMalloc(&d_input, h_input.size() * sizeof(float));
            cudaMalloc(&d_conv1_output, batch_size * hidden_channels * pool1_out_height * pool1_out_width * sizeof(float));
            cudaMalloc(&d_final_output, batch_size * out_channels * pool2_out_height * pool2_out_width * sizeof(float));

            // Copy input to device
            cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);

            // Forward pass
            conv1.forward(d_input, d_conv1_output, batch_size, height, width);
            conv2.forward(d_conv1_output, d_final_output, batch_size, pool1_out_height, pool1_out_width);

            // Get output for loss calculation
            std::vector<float> h_output(batch_size * out_channels * pool2_out_height * pool2_out_width);
            cudaMemcpy(h_output.data(), d_final_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

            // Calculate loss
            float loss = calculate_loss(h_output.data(), h_target.data(), h_output.size());
            std::cout << "Loss: " << loss << std::endl;

            // Print first batch input and output
            print_tensor(h_input.data(), in_channels, height, width, "Input (first batch)");
            print_tensor(h_output.data(), out_channels, pool2_out_height, pool2_out_width, 
                        "Output (first batch)");

            // Allocate memory for gradients
            float *d_grad_output, *d_grad_conv1_output, *d_grad_input;
            cudaMalloc(&d_grad_output, h_output.size() * sizeof(float));
            cudaMalloc(&d_grad_conv1_output, batch_size * hidden_channels * pool1_out_height * pool1_out_width * sizeof(float));
            cudaMalloc(&d_grad_input, h_input.size() * sizeof(float));

            // Simple gradient (difference from target)
            std::vector<float> h_grad_output(h_output.size());
            for (size_t i = 0; i < h_output.size(); i++) {
                h_grad_output[i] = h_output[i] - h_target[i];
            }
            cudaMemcpy(d_grad_output, h_grad_output.data(), h_grad_output.size() * sizeof(float), 
                      cudaMemcpyHostToDevice);

            // Backward pass
            conv2.backward(d_grad_output, d_grad_conv1_output, batch_size, pool1_out_height, pool1_out_width);
            conv1.backward(d_grad_conv1_output, d_grad_input, batch_size, height, width);

            // Get gradients and updated parameters
            std::vector<float> h_grad_input(h_input.size());
            cudaMemcpy(h_grad_input.data(), d_grad_input, h_grad_input.size() * sizeof(float), 
                      cudaMemcpyDeviceToHost);

            // Print gradients
            print_tensor(h_grad_input.data(), in_channels, height, width, 
                        "Gradient w.r.t Input (first batch)");

            // Print updated parameters
            std::cout << "\nUpdated Parameters after epoch " << epoch + 1 << ":";
            
            // Conv1 parameters
            std::vector<float> h_conv1_weights(hidden_channels * in_channels * kernel_size * kernel_size);
            std::vector<float> h_conv1_biases(hidden_channels);
            cudaMemcpy(h_conv1_weights.data(), conv1.get_weights(), 
                      h_conv1_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_conv1_biases.data(), conv1.get_biases(), 
                      h_conv1_biases.size() * sizeof(float), cudaMemcpyDeviceToHost);

            // Conv2 parameters
            std::vector<float> h_conv2_weights(out_channels * hidden_channels * kernel_size * kernel_size);
            std::vector<float> h_conv2_biases(out_channels);
            cudaMemcpy(h_conv2_weights.data(), conv2.get_weights(), 
                      h_conv2_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_conv2_biases.data(), conv2.get_biases(), 
                      h_conv2_biases.size() * sizeof(float), cudaMemcpyDeviceToHost);

            // Print parameter statistics
            std::cout << "\nConv1 Weights (mean): " 
                     << std::accumulate(h_conv1_weights.begin(), h_conv1_weights.end(), 0.0f) / h_conv1_weights.size();
            std::cout << "\nConv2 Weights (mean): " 
                     << std::accumulate(h_conv2_weights.begin(), h_conv2_weights.end(), 0.0f) / h_conv2_weights.size();

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

            std::cout << "\nEpoch " << epoch + 1 << " completed successfully!\n";
        }

        std::cout << "\nAll epochs completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}