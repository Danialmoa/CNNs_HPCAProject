#include "../include/conv_block.cuh"
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <cmath>

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

// CPU reference implementation
void cpu_convolution(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size,
    int stride, int padding, int out_height, int out_width) {
    
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float sum = biases[oc];
                    
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int h_in = oh * stride - padding + kh;
                                int w_in = ow * stride - padding + kw;
                                
                                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                                    int input_idx = ((b * in_channels + ic) * height + h_in) * width + w_in;
                                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                    sum += input[input_idx] * weights[weight_idx];
                                }
                            }
                        }
                    }
                    
                    output[((b * out_channels + oc) * out_height + oh) * out_width + ow] = sum;
                }
            }
        }
    }
}

void test_convolution_case(int batch_size, int in_channels, int out_channels, 
                          int height, int width, int kernel_size, int stride, 
                          int padding, std::ofstream& log_file) {
    log_file << "\n=== Testing Convolution ===" << std::endl;
    log_file << "Configuration:" << std::endl;
    log_file << "Batch size: " << batch_size << std::endl;
    log_file << "Input channels: " << in_channels << std::endl;
    log_file << "Output channels: " << out_channels << std::endl;
    log_file << "Input size: " << height << "x" << width << std::endl;
    log_file << "Kernel size: " << kernel_size << std::endl;
    log_file << "Stride: " << stride << std::endl;
    log_file << "Padding: " << padding << std::endl;

    // Calculate output dimensions
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;

    // Create input data with a recognizable pattern
    std::vector<float> h_input(batch_size * in_channels * height * width);
    for (size_t i = 0; i < h_input.size(); i++) {
        h_input[i] = static_cast<float>(i % 10);  // Simple repeating pattern
    }

    // Create weights and biases
    std::vector<float> h_weights(out_channels * in_channels * kernel_size * kernel_size);
    std::vector<float> h_biases(out_channels);
    for (size_t i = 0; i < h_weights.size(); i++) {
        h_weights[i] = 1.0f / static_cast<float>(h_weights.size());  // Small weights
    }
    for (size_t i = 0; i < h_biases.size(); i++) {
        h_biases[i] = 0.1f;  // Small bias
    }

    // Allocate GPU memory
    float *d_input, *d_weights, *d_biases, *d_output;
    cudaMalloc(&d_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_weights, h_weights.size() * sizeof(float));
    cudaMalloc(&d_biases, h_biases.size() * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * out_height * out_width * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, h_biases.data(), h_biases.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Create ConvBlock
    float learning_rate = 0.01;
    ConvBlock conv_block(in_channels, out_channels, kernel_size, 
                        stride, padding, 1, 1, learning_rate);  // No pooling

    // Print input tensor
    print_4d_tensor(h_input, batch_size, in_channels, height, width, "Input", log_file);

    // Forward pass
    conv_block.forward(d_input, d_output, batch_size, height, width);

    // Get GPU results
    std::vector<float> h_output_gpu(batch_size * out_channels * out_height * out_width);
    cudaMemcpy(h_output_gpu.data(), d_output, h_output_gpu.size() * sizeof(float), 
               cudaMemcpyDeviceToHost);

    // CPU reference computation
    std::vector<float> h_output_cpu(h_output_gpu.size());
    cpu_convolution(h_input.data(), h_weights.data(), h_biases.data(), h_output_cpu.data(),
                   batch_size, in_channels, out_channels, height, width,
                   kernel_size, stride, padding, out_height, out_width);

    // Compare results
    float max_diff = 0.0f;
    for (size_t i = 0; i < h_output_gpu.size(); i++) {
        float diff = std::abs(h_output_cpu[i] - h_output_gpu[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-4) {
            log_file << "Mismatch at index " << i << ": CPU=" << h_output_cpu[i] 
                    << ", GPU=" << h_output_gpu[i] << ", diff=" << diff << std::endl;
        }
    }

    log_file << "Maximum difference between CPU and GPU: " << max_diff << std::endl;
    
    // Print GPU output tensor
    print_4d_tensor(h_output_gpu, batch_size, out_channels, out_height, out_width, 
                    "GPU Output", log_file);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_output);
}

int main() {
    try {
        std::ofstream log_file("conv_test_detailed.txt");

        // Test case 1: Simple case
        test_convolution_case(1, 1, 1, 4, 4, 3, 1, 1, log_file);

        // Test case 2: Multiple channels
        test_convolution_case(1, 3, 2, 8, 8, 3, 1, 1, log_file);

        // Test case 3: Multiple batches
        test_convolution_case(2, 2, 2, 6, 6, 3, 1, 1, log_file);

        // Test case 4: Stride 2
        test_convolution_case(1, 1, 1, 8, 8, 3, 2, 1, log_file);

        // Test case 5: Larger kernel
        test_convolution_case(1, 1, 1, 8, 8, 5, 1, 2, log_file);

        log_file.close();
        std::cout << "Tests completed! Check conv_test_detailed.txt for results." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}