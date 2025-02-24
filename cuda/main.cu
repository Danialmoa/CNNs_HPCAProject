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

void check_tensor_values(const char* name, float* d_tensor, int size) {
    float* h_tensor = new float[size];
    cudaMemcpy(h_tensor, d_tensor, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float max_val = 10;
    float min_val = -10;
    float sum = 0.0f;
    bool has_nan = false;
    bool has_inf = false;
    
    for (int i = 0; i < size; i++) {
        if (isnan(h_tensor[i])) has_nan = true;
        if (isinf(h_tensor[i])) has_inf = true;
        if (!isnan(h_tensor[i]) && !isinf(h_tensor[i])) {
            max_val = max(max_val, h_tensor[i]);
            min_val = min(min_val, h_tensor[i]);
            sum += h_tensor[i];
        }
    }
    
    std::cout << name << " - Min: " << min_val << " Max: " << max_val 
              << " Sum: " << sum << " HasNaN: " << has_nan 
              << " HasInf: " << has_inf << std::endl;
    
    delete[] h_tensor;
}


int main() {
    try {
        // Set device to use
        CHECK_CUDA_ERROR(cudaSetDevice(0));
        
        cudaDeviceProp prop;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Using GPU: " << prop.name << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;

        // Training hyperparameters
        const std::vector<int> batch_sizes = {128};
        const int num_epochs = 1;
        const float learning_rate = 0.0001f;
        
        // Initialize dataset
        std::cout << "Initializing dataset..." << std::endl;
        DataSet dataset("../data");
        dataset.load_data();
        dataset.to_gpu();

        for (int batch_size : batch_sizes) {
            std::cout << "\n=== Testing Batch Size: " << batch_size << " ===" << std::endl;

            const int num_batches = dataset.get_num_batches(batch_size);
            std::cout << "Number of batches: " << num_batches << std::endl;
            
            // Create network layers
            std::cout << "Creating network..." << std::endl;
            ConvBlock conv1(3, 32, 3, 1, 1, 2, 2, learning_rate);  // input: 32x32x3, output: 16x16x32
            ConvBlock conv2(32, 64, 3, 1, 1, 2, 2, learning_rate);  // input: 16x16x32, output: 8x8x64
            ConvBlock conv3(64, 128, 3, 1, 1, 2, 2, learning_rate);  // input: 8x8x64, output: 4x4x128
            FullyConnectedLayer fc(128 * 4 * 4, 10, learning_rate*0.1f);  // input: 128x4x4, output: 10
            std::cout << "Network created" << std::endl;
            // Allocate GPU memory for data and intermediate results
            float *d_batch_images = nullptr;
            uint8_t *d_batch_labels = nullptr;
            float *d_conv_1_output = nullptr;
            float *d_conv_2_output = nullptr;
            float *d_conv_3_output = nullptr;
            float *d_fc_output = nullptr;

            float *d_grad_conv_1_output = nullptr;
            float *d_grad_conv_2_output = nullptr;
            float *d_grad_conv_3_output = nullptr;
            float *d_grad_fc_output = nullptr;

            const int conv1_output_size = batch_size * 32 * 16 * 16;
            const int conv2_output_size = batch_size * 64 * 8 * 8;
            const int conv3_output_size = batch_size * 128 * 4 * 4;

            const int fc_output_size = batch_size * 10;
            const int fc_grad_size = batch_size * 128 * 4 * 4;
            
       
            // Allocate memory
            CHECK_CUDA_ERROR(cudaMalloc(&d_batch_images, batch_size * 3 * 32 * 32 * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_batch_labels, batch_size * 10 * sizeof(uint8_t)));
            
            CHECK_CUDA_ERROR(cudaMalloc(&d_fc_output, fc_output_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_grad_fc_output, fc_grad_size * sizeof(float)));

            CHECK_CUDA_ERROR(cudaMalloc(&d_conv_1_output, conv1_output_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_conv_2_output, conv2_output_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_conv_3_output, conv3_output_size * sizeof(float)));

            CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv_1_output, conv1_output_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv_2_output, conv2_output_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv_3_output, conv3_output_size * sizeof(float)));
            
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
                    conv1.forward(d_batch_images, d_conv_1_output, batch_size, 32, 32);
                    conv2.forward(d_conv_1_output, d_conv_2_output, batch_size, 16, 16);
                    conv3.forward(d_conv_2_output, d_conv_3_output, batch_size, 8, 8);
                    fc.forward(d_conv_3_output, d_fc_output, batch_size);

                    check_tensor_values("d_fc_output", d_fc_output, 10);
                    // Compute loss and accuracy
                    float batch_loss = fc.compute_loss(d_batch_labels, batch_size);
                    float batch_accuracy = calculate_accuracy(d_fc_output, d_batch_labels, batch_size);
                    
                    epoch_loss += batch_loss;
                    epoch_accuracy += batch_accuracy;
                    std::cout << "Batch " << batch + 1 << "/" << num_batches << " - Loss: " << batch_loss << " - Accuracy: " << batch_accuracy * 100 << "%" << std::endl;

                    // Backward pass
                    fc.backward(d_batch_labels, d_grad_fc_output, batch_size);
                    conv3.backward(d_grad_fc_output, d_grad_conv_3_output, batch_size, 4, 4);
                    conv2.backward(d_grad_conv_3_output, d_grad_conv_2_output, batch_size, 8, 8);
                    conv1.backward(d_grad_conv_2_output, d_grad_conv_1_output, batch_size, 16, 16);
                    check_tensor_values("d_grad_conv_1_output", d_grad_conv_1_output, conv1_output_size);
                    // Synchronize streams
                    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

                }

                auto epoch_end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(epoch_end - epoch_start);

                epoch_loss /= num_batches;
                epoch_accuracy /= num_batches;

                std::cout << "\nEpoch " << epoch + 1 << "/" << num_epochs 
                        << " - Loss: " << epoch_loss 
                        << " - Accuracy: " << epoch_accuracy * 100 << "%" 
                        << " - Time: " << duration.count() << "ms" << std::endl;
                
            }

            // Free GPU memory
            cudaFree(d_batch_images);
            cudaFree(d_batch_labels);
            cudaFree(d_conv_1_output);
            cudaFree(d_conv_2_output);
            cudaFree(d_conv_3_output);
            cudaFree(d_fc_output);
            cudaFree(d_grad_conv_1_output);
            cudaFree(d_grad_conv_2_output);
            cudaFree(d_grad_conv_3_output);
            cudaFree(d_grad_fc_output);
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