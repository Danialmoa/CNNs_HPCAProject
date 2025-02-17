#include "../include/fully_connected.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

void test_fully_connected_layer() {
    // Test parameters
    const int batch_size = 2;
    const int in_features = 3;
    const int num_classes = 2;
    const float learning_rate = 0.001f;

    std::cout << "\n=== Testing FullyConnectedLayer ===\n" << std::endl;
    
    // Create test input data
    std::vector<float> h_input(batch_size * in_features);
    std::vector<uint8_t> h_labels(batch_size * num_classes, 0);
    
    // Initialize input with simple pattern
    for (int i = 0; i < h_input.size(); i++) {
        h_input[i] = static_cast<float>(i + 1);
    }
    
    // Set labels: first sample class 0, second sample class 1
    h_labels[0] = 1;  // First sample, class 0
    h_labels[batch_size + 1] = 1;  // Second sample, class 1
    
    // Allocate device memory
    float *d_input, *d_output;
    uint8_t *d_labels;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, batch_size * num_classes * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, h_labels.size() * sizeof(uint8_t)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), 
                               h_input.size() * sizeof(float), 
                               cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_labels.data(), 
                               h_labels.size() * sizeof(uint8_t), 
                               cudaMemcpyHostToDevice));
    
    try {
        // Create layer
        std::cout << "Creating FullyConnectedLayer..." << std::endl;
        FullyConnectedLayer layer(in_features, num_classes, learning_rate);
        
        // Forward pass
        std::cout << "Testing forward pass..." << std::endl;
        layer.forward(d_input, d_output, batch_size);
        
        // Get output
        std::vector<float> h_output(batch_size * num_classes);
        CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output,
                                   h_output.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        
        // Print output
        std::cout << "Forward pass output:" << std::endl;
        for (int b = 0; b < batch_size; b++) {
            std::cout << "Sample " << b << ": ";
            for (int c = 0; c < num_classes; c++) {
                std::cout << h_output[b * num_classes + c] << " ";
            }
            std::cout << std::endl;
        }
        
        // Compute loss
        float loss = layer.compute_loss(d_labels, batch_size);
        std::cout << "Initial loss: " << loss << std::endl;
        
        // Test backward pass
        std::cout << "\nTesting backward pass..." << std::endl;
        float* d_grad_input;
        CHECK_CUDA_ERROR(cudaMalloc(&d_grad_input, h_input.size() * sizeof(float)));
        
        // Perform multiple training iterations
        const int num_iterations = 5;
        for (int i = 0; i < num_iterations; i++) {
            layer.forward(d_input, d_output, batch_size);
            float iter_loss = layer.compute_loss(d_labels, batch_size);
            layer.backward(d_labels, d_grad_input, batch_size);
            
            std::cout << "Iteration " << i << " loss: " << iter_loss << std::endl;
            
            // Verify loss is decreasing
            if (i > 0 && iter_loss >= loss) {
                std::cout << "Warning: Loss not decreasing" << std::endl;
            }
            loss = iter_loss;
        }
        
        // Cleanup
        cudaFree(d_grad_input);
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        throw;
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_labels);
    
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
        test_fully_connected_layer();
        
        // Reset device
        CHECK_CUDA_ERROR(cudaDeviceReset());
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 