#include "../include/conv_block.cuh"
#include <iostream>
#include <vector>
#include <iomanip>
#include <numeric>
#include <random>
#include <cmath>

// Add function declarations at the top of the file
float calculate_mean(const float* data, int size);
float calculate_std(const float* data, int size);

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

// Helper function to generate random test input
float* generate_test_input(int batch_size, int channels, int height, int width) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    size_t size = batch_size * channels * height * width;
    float* data = new float[size];
    for (size_t i = 0; i < size; i++) {
        data[i] = dist(gen);
    }
    return data;
}

// Helper function to calculate mean and variance
void calculate_stats(const float* data, int size, float& mean, float& var) {
    mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += data[i];
    }
    mean /= size;

    var = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = data[i] - mean;
        var += diff * diff;
    }
    var /= size;
}

// Helper function to print tensor statistics
void print_tensor_stats(const float* data, int channels, int height, int width, const std::string& name) {
    float mean, var;
    int size = channels * height * width;
    calculate_stats(data, size, mean, var);
    std::cout << name << " stats - Mean: " << mean << ", Std: " << sqrt(var) << std::endl;
}

int main() {
    try {
        // Test parameters
        const int batch_size = 2;
        const int in_channels = 3;
        const int hidden_channels = 16;
        const int out_channels = 32;
        const int height = 32;
        const int width = 32;
        const int kernel_size = 3;
        const int stride = 1;
        const int padding = 1;
        const int pool_size = 2;
        const int pool_stride = 2;
        const float learning_rate = 0.001f;

        // Create ConvBlock instances
        ConvBlock conv1(in_channels, hidden_channels, kernel_size, stride, padding, 
                       pool_size, pool_stride, learning_rate);
        ConvBlock conv2(hidden_channels, out_channels, kernel_size, stride, padding, 
                       pool_size, pool_stride, learning_rate);

        // Calculate output dimensions
        int conv1_out_height = (height + 2 * padding - kernel_size) / stride + 1;
        int conv1_out_width = (width + 2 * padding - kernel_size) / stride + 1;
        int pool1_out_height = (conv1_out_height - pool_size) / pool_stride + 1;
        int pool1_out_width = (conv1_out_width - pool_size) / pool_stride + 1;

        int conv2_out_height = (pool1_out_height + 2 * padding - kernel_size) / stride + 1;
        int conv2_out_width = (pool1_out_width + 2 * padding - kernel_size) / stride + 1;
        int pool2_out_height = (conv2_out_height - pool_size) / pool_stride + 1;
        int pool2_out_width = (conv2_out_width - pool_size) / pool_stride + 1;

        // Generate input data
        float* h_input = generate_test_input(batch_size, in_channels, height, width);

        // Allocate device memory
        float *d_input, *d_conv1_output, *d_final_output;
        cudaMalloc(&d_input, batch_size * in_channels * height * width * sizeof(float));
        cudaMalloc(&d_conv1_output, batch_size * hidden_channels * pool1_out_height * pool1_out_width * sizeof(float));
        cudaMalloc(&d_final_output, batch_size * out_channels * pool2_out_height * pool2_out_width * sizeof(float));

        // Copy input to device
        cudaMemcpy(d_input, h_input, batch_size * in_channels * height * width * sizeof(float), 
                  cudaMemcpyHostToDevice);

        // Print input statistics
        print_tensor_stats(h_input, in_channels, height, width, "Input");

        // Print more detailed diagnostics
        std::cout << "\n=== Initial Weights Statistics ===\n";
        std::vector<float> h_weights1(hidden_channels * in_channels * kernel_size * kernel_size);
        cudaMemcpy(h_weights1.data(), conv1.get_weights(), h_weights1.size() * sizeof(float), 
                  cudaMemcpyDeviceToHost);
        print_tensor_stats(h_weights1.data(), hidden_channels, kernel_size, kernel_size, "Conv1 Weights");

        // Forward pass with intermediate checks
        conv1.forward(d_input, d_conv1_output, batch_size, height, width);
        
        // Check pre-batch norm values
        std::vector<float> h_conv1_pre_bn(batch_size * hidden_channels * conv1_out_height * conv1_out_width);
        cudaMemcpy(h_conv1_pre_bn.data(), conv1.get_conv_output_cache(), 
                  h_conv1_pre_bn.size() * sizeof(float), cudaMemcpyDeviceToHost);
        print_tensor_stats(h_conv1_pre_bn.data(), hidden_channels, conv1_out_height, conv1_out_width, 
                          "Conv1 Pre-BN Output");

        // Add synchronization point
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Check batch norm parameters with error checking
        std::vector<float> h_gamma(hidden_channels), h_beta(hidden_channels);
        std::vector<float> h_running_mean(hidden_channels), h_running_var(hidden_channels);
        
        CHECK_CUDA_ERROR(cudaMemcpy(h_gamma.data(), conv1.get_gamma(), 
                                   hidden_channels * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_beta.data(), conv1.get_beta(), 
                                   hidden_channels * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_running_mean.data(), conv1.get_running_mean(), 
                                   hidden_channels * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_running_var.data(), conv1.get_running_var(), 
                                   hidden_channels * sizeof(float), cudaMemcpyDeviceToHost));

        // Add debug prints
        std::cout << "\n=== Debug Information ===\n";
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Hidden channels: " << hidden_channels << std::endl;
        std::cout << "Conv1 output dimensions: " << conv1_out_height << "x" << conv1_out_width << std::endl;
        std::cout << "Pool1 output dimensions: " << pool1_out_height << "x" << pool1_out_width << std::endl;

        // Print intermediate values
        std::cout << "\n=== Batch Norm Parameters ===\n";
        std::cout << "Gamma (first few): ";
        for (int i = 0; i < std::min(5, hidden_channels); i++) {
            std::cout << h_gamma[i] << " ";
        }
        std::cout << "\nBeta (first few): ";
        for (int i = 0; i < std::min(5, hidden_channels); i++) {
            std::cout << h_beta[i] << " ";
        }
        std::cout << "\nRunning Mean (first few): ";
        for (int i = 0; i < std::min(5, hidden_channels); i++) {
            std::cout << h_running_mean[i] << " ";
        }
        std::cout << "\nRunning Var (first few): ";
        for (int i = 0; i < std::min(5, hidden_channels); i++) {
            std::cout << h_running_var[i] << " ";
        }
        std::cout << "\n";

        // Add synchronization point
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Forward pass for conv2
        conv2.forward(d_conv1_output, d_final_output, batch_size, pool1_out_height, pool1_out_width);

        // Add synchronization point
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Get intermediate output with error checking
        std::vector<float> h_conv1_output(batch_size * hidden_channels * pool1_out_height * pool1_out_width);
        CHECK_CUDA_ERROR(cudaMemcpy(h_conv1_output.data(), d_conv1_output, 
                                   h_conv1_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // Print detailed statistics for conv1 output
        float conv1_mean = 0.0f, conv1_var = 0.0f;
        calculate_stats(h_conv1_output.data(), h_conv1_output.size(), conv1_mean, conv1_var);
        
        std::cout << "\n=== Detailed Conv1 Output Statistics ===\n";
        std::cout << "Size: " << h_conv1_output.size() << std::endl;
        std::cout << "Mean: " << conv1_mean << std::endl;
        std::cout << "Variance: " << conv1_var << std::endl;
        std::cout << "Min: " << *std::min_element(h_conv1_output.begin(), h_conv1_output.end()) << std::endl;
        std::cout << "Max: " << *std::max_element(h_conv1_output.begin(), h_conv1_output.end()) << std::endl;

        // Get final output
        std::vector<float> h_final_output(batch_size * out_channels * pool2_out_height * pool2_out_width);
        cudaMemcpy(h_final_output.data(), d_final_output, h_final_output.size() * sizeof(float), 
                  cudaMemcpyDeviceToHost);

        // Print output statistics
        print_tensor_stats(h_conv1_output.data(), hidden_channels, pool1_out_height, pool1_out_width, 
                          "Conv1 Output");
        print_tensor_stats(h_final_output.data(), out_channels, pool2_out_height, pool2_out_width, 
                          "Final Output");

        // Verify batch normalization
        // After batch norm, we expect mean close to 0 and variance close to 1
        float conv1_mean_after, conv1_var_after;
        calculate_stats(h_conv1_output.data(), h_conv1_output.size(), conv1_mean_after, conv1_var_after);
        
        std::cout << "\nBatch Normalization Verification:";
        std::cout << "\nConv1 Output - Mean should be close to 0: " << conv1_mean_after;
        std::cout << "\nConv1 Output - Variance should be close to 1: " << conv1_var_after;

        // Test backward pass
        std::vector<float> h_grad_output(h_final_output.size(), 1.0f);  // Simple gradient
        float *d_grad_output, *d_grad_input;
        cudaMalloc(&d_grad_output, h_grad_output.size() * sizeof(float));
        cudaMalloc(&d_grad_input, batch_size * in_channels * height * width * sizeof(float));

        cudaMemcpy(d_grad_output, h_grad_output.data(), h_grad_output.size() * sizeof(float), 
                  cudaMemcpyHostToDevice);

        // Backward pass
        conv2.backward(d_grad_output, d_conv1_output, batch_size, pool1_out_height, pool1_out_width);
        conv1.backward(d_conv1_output, d_grad_input, batch_size, height, width);

        // Get gradients
        std::vector<float> h_grad_input(batch_size * in_channels * height * width);
        cudaMemcpy(h_grad_input.data(), d_grad_input, h_grad_input.size() * sizeof(float), 
                  cudaMemcpyDeviceToHost);

        // Print gradient statistics
        print_tensor_stats(h_grad_input.data(), in_channels, height, width, "Input Gradients");

        // For gradient testing, print more detailed gradient statistics
        std::cout << "\n=== Gradient Statistics ===\n";
        std::vector<float> h_grad_weights(hidden_channels * in_channels * kernel_size * kernel_size);
        cudaMemcpy(h_grad_weights.data(), conv1.get_grad_weights(), 
                  h_grad_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
        print_tensor_stats(h_grad_weights.data(), hidden_channels, kernel_size, kernel_size, 
                          "Conv1 Weight Gradients");

        // Cleanup with error checking
        delete[] h_input;
        CHECK_CUDA_ERROR(cudaFree(d_input));
        CHECK_CUDA_ERROR(cudaFree(d_conv1_output));
        CHECK_CUDA_ERROR(cudaFree(d_final_output));
        CHECK_CUDA_ERROR(cudaFree(d_grad_output));
        CHECK_CUDA_ERROR(cudaFree(d_grad_input));

        // Final synchronization
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(cudaGetLastError());

        std::cout << "\nTest completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// Function implementations remain at the bottom
float calculate_mean(const float* data, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

float calculate_std(const float* data, int size) {
    float mean = calculate_mean(data, size);
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / size);
}