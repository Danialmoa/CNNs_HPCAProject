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

void print_label(const std::vector<uint8_t>& labels, int image_idx) {
    std::cout << "Label " << image_idx << ": ";
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (labels[image_idx * NUM_CLASSES + i] == 1) {
            std::cout << "Class " << i << std::endl;
            break;
        }
    }
}

int main() {
    try {
        std::string data_path = "../../data";  // Adjust path as needed
        DataSet dataset(data_path);

        std::cout << "Loading data..." << std::endl;
        dataset.load_data();
        
        // Print initial data statistics (before GPU transfer)
        std::cout << "\nInitial data statistics:" << std::endl;
        const auto& images = dataset.get_images();
        const auto& labels = dataset.get_labels();
        
        // Print first few images stats
        for (int i = 0; i < 3; i++) {
            print_image_stats(images, i);
            print_label(labels, i);
            std::cout << std::endl;
        }

        // Transfer to GPU
        std::cout << "Transferring to GPU..." << std::endl;
        dataset.to_gpu();

        // Test batch retrieval
        int batch_size = 50;
        int total_batches = dataset.get_num_batches(batch_size);
        float* d_batch_images;
        uint8_t* d_batch_labels;
        
        cudaMalloc(&d_batch_images, batch_size * IMAGE_SIZE * sizeof(float));
        cudaMalloc(&d_batch_labels, batch_size * NUM_CLASSES * sizeof(uint8_t));

        std::cout << "\nTesting batches (total batches: " << total_batches << ")..." << std::endl;
        
        // Test first few batches
        for (int batch_idx = 0; batch_idx < total_batches; batch_idx++) {
            std::cout << "\nProcessing batch " << batch_idx << "/" << (total_batches - 1) 
                      << " (" << (batch_idx * 100.0f / total_batches) << "%)" << std::endl;
            
            dataset.get_batch_data(d_batch_images, d_batch_labels, batch_idx, batch_size);
            
            // Copy batch data back to CPU for verification
            std::vector<float> batch_images(batch_size * IMAGE_SIZE);
            std::vector<uint8_t> batch_labels(batch_size * NUM_CLASSES);
            
            cudaMemcpy(batch_images.data(), d_batch_images, 
                      batch_size * IMAGE_SIZE * sizeof(float), 
                      cudaMemcpyDeviceToHost);
            cudaMemcpy(batch_labels.data(), d_batch_labels,
                      batch_size * NUM_CLASSES * sizeof(uint8_t), 
                      cudaMemcpyDeviceToHost);

            // Print statistics for first few images in batch
            std::cout << "\nBatch " << batch_idx << " statistics:" << std::endl;
            for (int i = 0; i < std::min(3, batch_size); i++) {
                float min_val = 1000.0f, max_val = -1000.0f, sum = 0.0f;
                int offset = i * IMAGE_SIZE;
                
                for (int j = 0; j < IMAGE_SIZE; j++) {
                    float val = batch_images[offset + j];
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                    sum += val;
                }
                
                std::cout << "Image " << (batch_idx * batch_size + i) << ":" << std::endl;
                std::cout << "Min: " << min_val << ", Max: " << max_val;
                std::cout << ", Mean: " << sum / IMAGE_SIZE << std::endl;
                
                // Print label
                std::cout << "Label: ";
                for (int j = 0; j < NUM_CLASSES; j++) {
                    if (batch_labels[i * NUM_CLASSES + j] == 1) {
                        std::cout << "Class " << j << std::endl;
                        break;
                    }
                }
            }
            
            // Ensure GPU operations are complete
            cudaDeviceSynchronize();
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