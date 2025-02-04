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

// ... existing includes and helper functions ...

void verify_batch_consistency(const std::vector<float>& original_images,
                            const std::vector<std::vector<uint8_t>>& original_labels,
                            const std::vector<float>& batch_images,
                            const std::vector<uint8_t>& batch_labels,
                            int batch_idx, int batch_size) {
    
    for (int i = 0; i < batch_size; i++) {
        // Calculate offsets
        int orig_idx = batch_idx * batch_size + i;
        int batch_offset = i * IMAGE_SIZE;
        int orig_offset = orig_idx * IMAGE_SIZE;
        
        // Verify image data
        bool image_match = true;
        for (int j = 0; j < IMAGE_SIZE; j++) {
            if (batch_images[batch_offset + j] != original_images[orig_offset + j]) {
                image_match = false;
                break;
            }
        }
        
        // Verify label
        bool label_match = true;
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (batch_labels[i * NUM_CLASSES + j] != original_labels[orig_idx][j]) {
                label_match = false;
                break;
            }
        }
        
        // Print verification results
        std::cout << "Image " << orig_idx << " verification: " 
                  << (image_match ? "PASS" : "FAIL") << std::endl;
        std::cout << "Label " << orig_idx << " verification: " 
                  << (label_match ? "PASS" : "FAIL") << std::endl;
    }
}

int main() {
    try {
        std::string data_path = "../../data";  // Adjust path as needed
        DataSet dataset(data_path);

        std::cout << "Loading data..." << std::endl;
        dataset.load_data();
        
        // Store original data for verification
        const auto& original_images = dataset.get_images();
        const auto& original_labels = dataset.get_labels();
        
        // Transfer to GPU
        std::cout << "Transferring to GPU..." << std::endl;
        dataset.to_gpu();

        // Test batch retrieval
        int batch_size = 32;
        int total_batches = dataset.get_num_batches(batch_size);
        float* d_batch_images;
        uint8_t* d_batch_labels;
        
        cudaMalloc(&d_batch_images, batch_size * IMAGE_SIZE * sizeof(float));
        cudaMalloc(&d_batch_labels, batch_size * NUM_CLASSES * sizeof(uint8_t));

        std::cout << "\nTesting all " << total_batches << " batches..." << std::endl;
        
        // Test all batches
        for (int batch_idx = 0; batch_idx < total_batches; batch_idx++) {
            std::cout << "\nProcessing batch " << batch_idx << "/" << total_batches - 1 << std::endl;
            
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

            // Verify batch data
            verify_batch_consistency(original_images, original_labels,
                                  batch_images, batch_labels,
                                  batch_idx, batch_size);

            // Print batch statistics
            std::cout << "\nBatch " << batch_idx << " statistics:" << std::endl;
            for (int i = 0; i < std::min(3, batch_size); i++) {  // Print first 3 images in batch
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
        }

        // Cleanup
        cudaFree(d_batch_images);
        cudaFree(d_batch_labels);
        
        std::cout << "\nAll batch tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}