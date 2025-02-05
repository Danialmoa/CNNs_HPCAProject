#include "../include/conv_block.cuh"
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <iomanip>

// Helper function to print 4D tensor (batch, channel, height, width)
void print_4d_tensor(const std::vector<float>& tensor, int batch_size, int channels, 
                    int height, int width, const std::string& name, std::ofstream& log_file) {
    log_file << "\n" << name << " shape: [" << batch_size << ", " << channels << ", " 
             << height << ", " << width << "]" << std::endl;
    
    for (int b = 0; b < batch_size; b++) {
        log_file << "Batch " << b << ":\n";
        for (int c = 0; c < channels; c++) {
            log_file << "Channel " << c << ":\n";
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = b * (channels * height * width) + 
                             c * (height * width) + 
                             h * width + w;
                    log_file << std::setw(8) << std::fixed << std::setprecision(4) 
                            << tensor[idx] << " ";
                }
                log_file << "\n";
            }
            log_file << "\n";
        }
        log_file << "\n";
    }
}

void test_simple_convolution() {
    std::ofstream log_file("conv_test_detailed.txt");
    log_file << "=== Simple Convolution Test ===" << std::endl;
    
    // Create a minimal ConvBlock (no pooling)
    int in_channels = 1;
    int out_channels = 1;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    int pool_size = 1;  // No pooling
    int pool_stride = 1;
    float learning_rate = 0.01;
    
    ConvBlock conv_block(in_channels, out_channels, kernel_size, 
                        stride, padding, pool_size, pool_stride, learning_rate);
    
    // Create a simple 4x4 input with a clear pattern
    int batch_size = 1;
    int height = 4;
    int width = 4;
    std::vector<float> h_input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    
    // Allocate GPU memory
    float *d_input, *d_output;
    int output_height = height;  // Same as input due to padding
    int output_width = width;    // Same as input due to padding
    
    cudaMalloc(&d_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * output_height * output_width * sizeof(float));
    
    // Copy input to GPU
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Print input
    print_4d_tensor(h_input, batch_size, in_channels, height, width, "Input", log_file);
    
    // Forward pass
    conv_block.forward(d_input, d_output, batch_size, height, width);
    
    // Copy output back to CPU
    std::vector<float> h_output(batch_size * out_channels * output_height * output_width);
    cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print output
    print_4d_tensor(h_output, batch_size, out_channels, output_height, output_width, "Output", log_file);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    log_file.close();
}

int main() {
    try {
        test_simple_convolution();
        std::cout << "Test completed! Check conv_test_detailed.txt for results." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}