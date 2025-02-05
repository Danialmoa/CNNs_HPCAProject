#include "../include/fully_connected.cuh"
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <iomanip>

// Helper function to print 2D tensor (batch_size x features)
void print_2d_tensor(const std::vector<float>& tensor, int batch_size, 
                    int features, const std::string& name, std::ofstream& log_file) {
    log_file << "\n" << name << " shape: [" << batch_size << ", " << features << "]" << std::endl;
    
    for (int b = 0; b < batch_size; b++) {
        log_file << "Batch " << b << ":\n";
        for (int f = 0; f < features; f++) {
            log_file << std::setw(8) << std::fixed << std::setprecision(4) 
                    << tensor[b * features + f] << " ";
            if ((f + 1) % 8 == 0) log_file << "\n";  // Line break every 8 numbers
        }
        log_file << "\n\n";
    }
}

void test_simple_forward() {
    std::ofstream log_file("fc_test_detailed.txt");
    log_file << "=== Simple Forward Pass Test ===" << std::endl;
    
    // Create a small fully connected layer
    int in_features = 4;
    int num_classes = 3;
    float learning_rate = 0.01f;
    
    FullyConnectedLayer fc_layer(in_features, num_classes, learning_rate);
    
    // Create a simple batch of inputs
    int batch_size = 2;
    std::vector<float> h_input = {
        // Batch 1
        1.0f, 2.0f, 3.0f, 4.0f,
        // Batch 2
        0.5f, 1.5f, 2.5f, 3.5f
    };
    
    // Allocate GPU memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_output, batch_size * num_classes * sizeof(float));
    
    // Copy input to GPU
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Print input
    print_2d_tensor(h_input, batch_size, in_features, "Input", log_file);
    
    // Forward pass
    fc_layer.forward(d_input, d_output, batch_size);
    
    // Copy output back to CPU
    std::vector<float> h_output(batch_size * num_classes);
    cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print output (softmax probabilities)
    print_2d_tensor(h_output, batch_size, num_classes, "Softmax Output", log_file);
    
    // Verify softmax properties
    log_file << "\nVerifying softmax properties:" << std::endl;
    for (int b = 0; b < batch_size; b++) {
        float sum = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum += h_output[b * num_classes + c];
        }
        log_file << "Batch " << b << " sum: " << sum << " (should be close to 1.0)" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

void test_backward() {
    std::ofstream log_file("fc_test_detailed.txt", std::ios::app);
    log_file << "\n=== Backward Pass Test ===" << std::endl;
    
    int in_features = 4;
    int num_classes = 3;
    float learning_rate = 0.01f;
    
    FullyConnectedLayer fc_layer(in_features, num_classes, learning_rate);
    
    int batch_size = 2;
    std::vector<float> h_input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        0.5f, 1.5f, 2.5f, 3.5f
    };
    
    // Create labels (one-hot encoded)
    std::vector<uint8_t> h_labels = {
        1, 0, 0,  // First batch item: class 0
        0, 1, 0   // Second batch item: class 1
    };
    
    // Allocate GPU memory
    float *d_input, *d_output, *d_grad_input;
    uint8_t *d_labels;
    
    cudaMalloc(&d_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_output, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_grad_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_labels, h_labels.size() * sizeof(uint8_t));
    
    // Copy data to GPU
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    // Forward pass
    fc_layer.forward(d_input, d_output, batch_size);
    
    // Compute loss
    float loss = fc_layer.compute_loss(d_labels, batch_size);
    log_file << "Initial loss: " << loss << std::endl;
    
    // Backward pass
    fc_layer.backward(d_labels, d_grad_input, batch_size);
    
    // Copy gradients back to CPU
    std::vector<float> h_grad_input(h_input.size());
    cudaMemcpy(h_grad_input.data(), d_grad_input, h_grad_input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print gradients
    print_2d_tensor(h_grad_input, batch_size, in_features, "Input Gradients", log_file);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_grad_input);
    cudaFree(d_labels);
}

void test_training_sequence() {
    std::ofstream log_file("fc_test_detailed.txt", std::ios::app);
    log_file << "\n=== Training Sequence Test ===" << std::endl;
    
    int in_features = 4;
    int num_classes = 3;
    float learning_rate = 0.01f;
    
    FullyConnectedLayer fc_layer(in_features, num_classes, learning_rate);
    
    int batch_size = 2;
    std::vector<float> h_input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        0.5f, 1.5f, 2.5f, 3.5f
    };
    
    std::vector<uint8_t> h_labels = {
        1, 0, 0,
        0, 1, 0
    };
    
    // Allocate GPU memory
    float *d_input, *d_output, *d_grad_input;
    uint8_t *d_labels;
    
    cudaMalloc(&d_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_output, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_grad_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_labels, h_labels.size() * sizeof(uint8_t));
    
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    // Training loop
    const int num_iterations = 10;
    log_file << "Training for " << num_iterations << " iterations:" << std::endl;
    
    for (int i = 0; i < num_iterations; i++) {
        // Forward pass
        fc_layer.forward(d_input, d_output, batch_size);
        
        // Compute and log loss
        float loss = fc_layer.compute_loss(d_labels, batch_size);
        log_file << "Iteration " << i << ", Loss: " << loss << std::endl;
        
        // Backward pass
        fc_layer.backward(d_labels, d_grad_input, batch_size);
    }
    
    // Final forward pass to check results
    fc_layer.forward(d_input, d_output, batch_size);
    std::vector<float> h_final_output(batch_size * num_classes);
    cudaMemcpy(h_final_output.data(), d_output, h_final_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    print_2d_tensor(h_final_output, batch_size, num_classes, "Final Output", log_file);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_grad_input);
    cudaFree(d_labels);
}

int main() {
    try {
        test_simple_forward();
        test_backward();
        test_training_sequence();
        
        std::cout << "All tests completed! Check fc_test_detailed.txt for results." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}