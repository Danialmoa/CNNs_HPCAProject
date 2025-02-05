#include "../include/adam_optimizer.cuh"
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

// Test initialization and basic update
void test_simple_update() {
    std::cout << "\nTesting simple parameter update..." << std::endl;
    
    // Create Adam optimizer with default hyperparameters
    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    
    AdamOptimizer optimizer(learning_rate, beta1, beta2, epsilon);
    
    // Create test parameters and gradients
    size_t num_params = 1000;
    std::vector<float> h_params(num_params, 1.0f);
    std::vector<float> h_gradients(num_params, 0.1f);
    
    // Allocate GPU memory
    float *d_params, *d_gradients;
    cudaMalloc(&d_params, num_params * sizeof(float));
    cudaMalloc(&d_gradients, num_params * sizeof(float));
    
    // Copy data to GPU
    cudaMemcpy(d_params, h_params.data(), num_params * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradients, h_gradients.data(), num_params * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize optimizer
    optimizer.init(num_params);
    
    // Perform update
    optimizer.update(d_params, d_gradients);
    
    // Copy parameters back to CPU
    cudaMemcpy(h_params.data(), d_params, num_params * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print statistics
    print_tensor_stats(h_params, "Updated Parameters");
    
    // Cleanup
    cudaFree(d_params);
    cudaFree(d_gradients);
}

// Test multiple updates with varying gradients
void test_training_sequence() {
    std::cout << "\nTesting training sequence..." << std::endl;
    
    AdamOptimizer optimizer(0.001f, 0.9f, 0.999f, 1e-8f);
    size_t num_params = 1000;
    
    // Allocate GPU memory
    float *d_params, *d_gradients;
    cudaMalloc(&d_params, num_params * sizeof(float));
    cudaMalloc(&d_gradients, num_params * sizeof(float));
    
    // Initialize parameters and optimizer
    std::vector<float> h_params(num_params, 1.0f);
    cudaMemcpy(d_params, h_params.data(), num_params * sizeof(float), cudaMemcpyHostToDevice);
    optimizer.init(num_params);
    
    // Perform multiple updates with different gradients
    for (int i = 0; i < 10; i++) {
        // Create gradients that decrease over time
        std::vector<float> h_gradients(num_params, 0.1f / (i + 1));
        cudaMemcpy(d_gradients, h_gradients.data(), num_params * sizeof(float), cudaMemcpyHostToDevice);
        
        // Update parameters
        optimizer.update(d_params, d_gradients);
        
        // Copy parameters back to check progress
        cudaMemcpy(h_params.data(), d_params, num_params * sizeof(float), cudaMemcpyDeviceToHost);
        
        std::cout << "Iteration " << i << ":" << std::endl;
        print_tensor_stats(h_params, "Parameters");
    }
    
    // Cleanup
    cudaFree(d_params);
    cudaFree(d_gradients);
}

// Test edge cases and error handling
void test_edge_cases() {
    std::cout << "\nTesting edge cases..." << std::endl;
    
    AdamOptimizer optimizer(0.001f, 0.9f, 0.999f, 1e-8f);
    
    try {
        // Test update before initialization
        float *d_params = nullptr, *d_gradients = nullptr;
        cudaMalloc(&d_params, 100 * sizeof(float));
        cudaMalloc(&d_gradients, 100 * sizeof(float));
        
        std::cout << "Testing update before initialization (should throw)..." << std::endl;
        optimizer.update(d_params, d_gradients);
        
        cudaFree(d_params);
        cudaFree(d_gradients);
    } catch (const std::runtime_error& e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
    }
    
    try {
        // Test with zero learning rate
        AdamOptimizer zero_lr_optimizer(0.0f, 0.9f, 0.999f, 1e-8f);
        std::cout << "Created optimizer with zero learning rate" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught error: " << e.what() << std::endl;
    }
}

int main() {
    try {
        // Run all tests
        test_simple_update();
        test_training_sequence();
        test_edge_cases();
        
        std::cout << "\nAll tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}