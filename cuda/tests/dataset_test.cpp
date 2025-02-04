#include "../include/dataset.cuh"
#include <iostream>
#include <iomanip>

void print_image_stats(const std::vector<float>& images, int image_idx) {
    float min_val = 1000.0f, max_val = -1000.0f, sum = 0.0f;
    int offset = image_idx * IMAGE_SIZE;
    
    for (int i = 0; i < IMAGE_SIZE; i++) {
        float val = images[offset + i];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
    }
    
    std::cout << "Image " << image_idx << " stats:" << std::endl;
    std::cout << "Min: " << min_val << ", Max: " << max_val;
    std::cout << ", Mean: " << sum / IMAGE_SIZE << std::endl;
}

void print_label(const std::vector<std::vector<uint8_t>>& labels, int image_idx) {
    std::cout << "Label " << image_idx << ": ";
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (labels[image_idx][i] == 1) {
            std::cout << "Class " << i << std::endl;
            break;
        }
    }
}

int main() {
    try {
        // Initialize dataset with path to CIFAR-10 binary files
        std::string data_path = "./cifar-10-batches-bin";  // Adjust this path
        DataSet dataset(data_path);

        std::cout << "Loading data..." << std::endl;
        dataset.load_data();
        
        // Print statistics for first few images before GPU transfer
        std::cout << "\nData statistics before GPU transfer:\n";
        for (int i = 0; i < 3; i++) {
            print_image_stats(dataset.h_images, i);
            print_label(dataset.h_labels, i);
            std::cout << std::endl;
        }

        // Transfer to GPU
        std::cout << "Transferring to GPU..." << std::endl;
        dataset.to_gpu();

        // Test batch retrieval
        int batch_size = 32;
        float* d_batch_images;
        uint8_t* d_batch_labels;
        
        // Allocate GPU memory for batch
        cudaMalloc(&d_batch_images, batch_size * IMAGE_SIZE * sizeof(float));
        cudaMalloc(&d_batch_labels, batch_size * NUM_CLASSES * sizeof(uint8_t));

        // Test getting different batches
        for (int batch_idx = 0; batch_idx < 3; batch_idx++) {
            std::cout << "\nRetrieving batch " << batch_idx << "..." << std::endl;
            dataset.get_batch_data(d_batch_images, d_batch_labels, batch_idx, batch_size);
            
            // Verify the data by copying back to CPU
            std::vector<float> batch_images(batch_size * IMAGE_SIZE);
            std::vector<uint8_t> batch_labels(batch_size * NUM_CLASSES);
            
            cudaMemcpy(batch_images.data(), d_batch_images, 
                      batch_size * IMAGE_SIZE * sizeof(float), 
                      cudaMemcpyDeviceToHost);
            cudaMemcpy(batch_labels.data(), d_batch_labels,
                      batch_size * NUM_CLASSES * sizeof(uint8_t), 
                      cudaMemcpyDeviceToHost);

            // Print first image stats from each batch
            float min_val = 1000.0f, max_val = -1000.0f, sum = 0.0f;
            for (int i = 0; i < IMAGE_SIZE; i++) {
                float val = batch_images[i];
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                sum += val;
            }
            
            std::cout << "Batch " << batch_idx << " first image stats:" << std::endl;
            std::cout << "Min: " << min_val << ", Max: " << max_val;
            std::cout << ", Mean: " << sum / IMAGE_SIZE << std::endl;
            
            // Print first label from batch
            std::cout << "First label in batch: ";
            for (int i = 0; i < NUM_CLASSES; i++) {
                if (batch_labels[i] == 1) {
                    std::cout << "Class " << i << std::endl;
                    break;
                }
            }
        }

        // Cleanup
        cudaFree(d_batch_images);
        cudaFree(d_batch_labels);
        
        std::cout << "\nTest completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}