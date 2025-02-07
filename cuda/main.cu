#include <iostream>
#include <chrono>
#include "include/dataset.cuh"
#include "include/conv_block.cuh"
#include "include/fully_connected.cuh"
#include "include/adam_optimizer.cuh"



void print_memory_requirements(int batch_size) {
    size_t total_memory = 0;
    
    // Calculate memory for each buffer
    size_t images_memory = batch_size * 3 * 32 * 32 * sizeof(float);
    size_t labels_memory = batch_size * 10 * sizeof(uint8_t);
    size_t conv_output_memory = batch_size * 32 * 16 * 16 * sizeof(float);
    size_t fc_output_memory = batch_size * 10 * sizeof(float);
    size_t grad_memory = images_memory + conv_output_memory;
    
    total_memory = images_memory + labels_memory + conv_output_memory + 
                   fc_output_memory + grad_memory;
    
    std::cout << "Memory requirements for batch size " << batch_size << ":" << std::endl;
    std::cout << "Images: " << (images_memory / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Labels: " << (labels_memory / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Conv output: " << (conv_output_memory / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "FC output: " << (fc_output_memory / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Gradients: " << (grad_memory / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Total: " << (total_memory / 1024.0 / 1024.0) << " MB" << std::endl;
}

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

        // Print available GPU memory before starting
        size_t free_byte, total_byte;
        CHECK_CUDA_ERROR(cudaMemGetInfo(&free_byte, &total_byte));
        std::cout << "GPU Memory before allocation:" << std::endl;
        std::cout << "Free: " << (free_byte / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "Total: " << (total_byte / 1024.0 / 1024.0) << " MB" << std::endl;

        // Training hyperparameters
        const int batch_size = 32;
        const int num_epochs = 10;
        const float learning_rate = 0.001f;
        print_memory_requirements(batch_size);
        
        // Initialize dataset
        std::cout << "Initializing dataset..." << std::endl;
        DataSet dataset("../../data");
        dataset.load_data();
        dataset.to_gpu();
        const int num_batches = dataset.get_num_batches(batch_size);
        std::cout << "Number of batches: " << num_batches << std::endl;
        
        // Create network layers
        std::cout << "Creating network..." << std::endl;
        ConvBlock conv1(3, 32, 3, 1, 1, 2, 2, learning_rate);  // input: 32x32x3, output: 16x16x32
        FullyConnectedLayer fc(32 * 16 * 16, 10, learning_rate);  // input: 8192, output: 10

        // Allocate GPU memory for data and intermediate results
        float *d_batch_images;
        uint8_t *d_batch_labels;
        float *d_conv_output, *d_fc_output;
        float *d_grad_conv_output, *d_grad_input;

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
                std::cout << "Batch " << batch << " of " << num_batches << std::endl;

                // Clear CUDA cache before each batch
                CHECK_CUDA_ERROR(cudaDeviceSynchronize());

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

                std::cout << "Batch loss: " << batch_loss << std::endl;
                std::cout << "Batch accuracy: " << batch_accuracy << std::endl;
                
                // Backward pass
                fc.backward(d_batch_labels, d_grad_conv_output, batch_size);
                conv1.backward(d_grad_conv_output, d_grad_input, batch_size);

                // Print progress every 10 batches
                if ((batch + 1) % 10 == 0) {
                    std::cout << "\rBatch " << batch + 1 << "/" << num_batches 
                              << " - Loss: " << batch_loss 
                              << " - Accuracy: " << batch_accuracy * 100 << "%" 
                              << std::flush;
                }
                CHECK_CUDA_ERROR(cudaDeviceSynchronize());
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

        // Clean up
        std::cout << "Cleaning up..." << std::endl;
        CHECK_CUDA_ERROR(cudaFree(d_batch_images));
        CHECK_CUDA_ERROR(cudaFree(d_batch_labels));
        CHECK_CUDA_ERROR(cudaFree(d_conv_output));
        CHECK_CUDA_ERROR(cudaFree(d_fc_output));
        CHECK_CUDA_ERROR(cudaFree(d_grad_conv_output));
        CHECK_CUDA_ERROR(cudaFree(d_grad_input));

        // Reset device
        CHECK_CUDA_ERROR(cudaDeviceReset());

        std::cout << "Training completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}