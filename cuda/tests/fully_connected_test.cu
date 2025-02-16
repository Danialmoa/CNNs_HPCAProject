#include "../include/fully_connected.cuh"
#include <iostream>
#include <vector>
#include <iomanip>
#include <numeric>

// Helper function to print 2D tensor (batch_size, features)
void print_tensor(const float* data, int batch_size, int features, const std::string& name) {
    std::cout << "\n" << name << " [" << batch_size << ", " << features << "]:\n";
    for (int b = 0; b < batch_size; b++) {
        std::cout << "Batch " << b << ": ";
        for (int f = 0; f < features; f++) {
            int idx = b * features + f;
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) 
                     << data[idx] << " ";
        }
        std::cout << "\n";
    }
}

// Simple MSE loss calculation
float calculate_loss(const float* output, const uint8_t* labels, int batch_size, int num_classes) {
    float loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < num_classes; c++) {
            float diff = output[b * num_classes + c] - labels[b * num_classes + c];
            loss += diff * diff;
        }
    }
    return loss / batch_size;
}

int main() {
    try {
        // Test parameters
        const int batch_size = 4;
        const int in_features = 32;    // Input features
        const int num_classes = 10;     // Output classes
        const float learning_rate = 0.01f;
        const int num_epochs = 2;

        // Create FullyConnectedLayer
        FullyConnectedLayer fc(in_features, num_classes, learning_rate);

        std::cout << "\nNetwork Architecture:";
        std::cout << "\nInput: " << batch_size << "x" << in_features;
        std::cout << "\nOutput: " << batch_size << "x" << num_classes;
        std::cout << "\n";

        // Training loop
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::cout << "\n=== Epoch " << epoch + 1 << " ===\n";

            // Create input data
            std::vector<float> h_input(batch_size * in_features);
            for (int b = 0; b < batch_size; b++) {
                for (int f = 0; f < in_features; f++) {
                    int idx = b * in_features + f;
                    h_input[idx] = (float)(f % 5) * 0.1f + epoch * 0.01f;  // Simple pattern
                }
            }

            // Create target labels (one-hot encoded)
            std::vector<uint8_t> h_labels(batch_size * num_classes, 0);
            for (int b = 0; b < batch_size; b++) {
                int target_class = b % num_classes;  // Assign different target for each batch
                h_labels[b * num_classes + target_class] = 1;
            }

            // Allocate device memory
            float *d_input, *d_output;
            uint8_t *d_labels;
            cudaMalloc(&d_input, h_input.size() * sizeof(float));
            cudaMalloc(&d_output, batch_size * num_classes * sizeof(float));
            cudaMalloc(&d_labels, h_labels.size() * sizeof(uint8_t));

            // Copy data to device
            cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), 
                      cudaMemcpyHostToDevice);
            cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(uint8_t), 
                      cudaMemcpyHostToDevice);

            // Forward pass
            fc.forward(d_input, d_output, batch_size);

            // Get output
            std::vector<float> h_output(batch_size * num_classes);
            cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), 
                      cudaMemcpyDeviceToHost);

            // Calculate and print loss
            float loss = fc.compute_loss(d_labels, batch_size);
            std::cout << "Loss: " << loss << std::endl;

            // Print input and output
            print_tensor(h_input.data(), batch_size, in_features, "Input");
            print_tensor(h_output.data(), batch_size, num_classes, "Output (Softmax probabilities)");

            // Allocate memory for gradients
            float *d_grad_input;
            cudaMalloc(&d_grad_input, h_input.size() * sizeof(float));

            // Backward pass
            fc.backward(d_labels, d_grad_input, batch_size);

            // Get gradients
            std::vector<float> h_grad_input(h_input.size());
            cudaMemcpy(h_grad_input.data(), d_grad_input, h_grad_input.size() * sizeof(float), 
                      cudaMemcpyDeviceToHost);

            // Print gradients
            print_tensor(h_grad_input.data(), batch_size, in_features, 
                        "Gradient w.r.t Input");

            // Cleanup iteration-specific memory
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_labels);
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