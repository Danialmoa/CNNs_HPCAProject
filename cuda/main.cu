#include <iostream>
#include <chrono>
#include "include/dataset.cuh"
#include "include/conv_block.cuh"
#include "include/fully_connected.cuh"
#include "include/adam_optimizer.cuh"



// Helper function to calculate accuracy
__global__ void calculate_accuracy_kernel(
    const float* predictions, const uint8_t* labels,
    int* correct_predictions, int batch_size, int num_classes) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float max_val = predictions[idx * num_classes];
    int pred_class = 0;

    // Find predicted class
    for (int i = 1; i < num_classes; i++) {
        float val = predictions[idx * num_classes + i];
        if (val > max_val) {
            max_val = val;
            pred_class = i;
        }
    }

    // Find true class
    int true_class = -1;
    for (int i = 0; i < num_classes; i++) {
        if (labels[idx * num_classes + i] == 1) {
            true_class = i;
            break;
        }
    }

    if (true_class != -1 && pred_class == true_class) {
        atomicAdd(correct_predictions, 1);
    }
}

float calculate_accuracy(const float* d_predictions, const uint8_t* d_labels, int batch_size) {
    int* d_correct;
    int h_correct;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_correct, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_correct, 0, sizeof(int)));

    int threadsPerBlock = 256;
    int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    
    calculate_accuracy_kernel<<<numBlocks, threadsPerBlock>>>(
        d_predictions, d_labels, d_correct, batch_size, 10
    );
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_correct);

    return static_cast<float>(h_correct) / batch_size;
}

int main() {
    try {
        // Set device to use
        CHECK_CUDA_ERROR(cudaSetDevice(0));


        // Training hyperparameters
        const int batch_size = 64;
        const int num_epochs = 10;
        const float learning_rate = 0.0005f;
        
        // Initialize dataset
        std::cout << "Initializing dataset..." << std::endl;
        DataSet dataset("../data");
        dataset.load_data();
        dataset.to_gpu();

        const int num_batches = dataset.get_num_batches(batch_size);
        std::cout << "Number of batches: " << num_batches << std::endl;
        
        // Create network layers
        std::cout << "Creating network..." << std::endl;
        ConvBlock conv1(3, 32, 3, 1, 1, 2, 2, learning_rate);  // input: 32x32x3, output: 16x16x32
        FullyConnectedLayer fc(32 * 16 * 16, 10, learning_rate);  // input: 8192, output: 10

        // Allocate GPU memory for data and intermediate results
        float *d_batch_images = nullptr;
        uint8_t *d_batch_labels = nullptr;
        float *d_conv_output = nullptr;
        float *d_fc_output = nullptr;
        float *d_grad_conv_output = nullptr;
        float *d_grad_input = nullptr;

        // Allocate memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_batch_images, batch_size * 3 * 32 * 32 * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_batch_labels, batch_size * 10 * sizeof(uint8_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_conv_output, batch_size * 32 * 16 * 16 * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_fc_output, batch_size * 10 * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv_output, batch_size * 32 * 16 * 16 * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_grad_input, batch_size * 3 * 32 * 32 * sizeof(float)));
        
        
        // Training loop
        std::cout << "Starting training..." << std::endl;
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            float epoch_loss = 0.0f;
            float epoch_accuracy = 0.0f;
            auto epoch_start = std::chrono::high_resolution_clock::now();

            for (int batch = 0; batch < num_batches; ++batch) {
                // Get batch data
                dataset.get_batch_data(d_batch_images, d_batch_labels, batch, batch_size);

                // Forward pass
                conv1.forward(d_batch_images, d_conv_output, batch_size, 32, 32);
                fc.forward(d_conv_output, d_fc_output, batch_size);

                // Compute loss and accuracy
                float batch_loss = fc.compute_loss(d_batch_labels, batch_size);
                float batch_accuracy = calculate_accuracy(d_fc_output, d_batch_labels, batch_size);

                epoch_loss += batch_loss;
                epoch_accuracy += batch_accuracy;
                
                // Backward pass
                fc.backward(d_batch_labels, d_grad_conv_output, batch_size);
                conv1.backward(d_grad_conv_output, d_grad_input, batch_size);

            }

            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);

            epoch_loss /= num_batches;
            epoch_accuracy /= num_batches;

            std::cout << "\nEpoch " << epoch + 1 << "/" << num_epochs 
                      << " - Loss: " << epoch_loss 
                      << " - Accuracy: " << epoch_accuracy * 100 << "%" 
                      << " - Time: " << duration.count() << "s" << std::endl;
            
        }

        // Reset device
        CHECK_CUDA_ERROR(cudaDeviceReset());

        std::cout << "Training completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}