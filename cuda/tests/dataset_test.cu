#include "../include/dataset.cuh"
#include <iostream>
#include <iomanip>

void print_image(const float* image_data, const uint8_t* label_data, int num_classes) {
    // Print label
    std::cout << "Label one-hot encoding: ";
    for (int i = 0; i < num_classes; i++) {
        std::cout << (int)label_data[i] << " ";  
    }
    std::cout << "\nLabel index: ";
    for (int i = 0; i < num_classes; i++) {
        if (label_data[i] == 1) {
            std::cout << i;
            break;
        }
    }
    std::cout << "\n\nImage data:\n";

    // Print each channel
    for (int c = 0; c < 3; c++) {
        std::cout << "\nChannel " << c << ":\n";
        for (int h = 0; h < 32; h++) {
            for (int w = 0; w < 32; w++) {
                int idx = (c * 32 * 32) + (h * 32) + w;
                // Print with 3 decimal places and fixed width
                std::cout << std::fixed << std::setprecision(3) << std::setw(6) 
                         << image_data[idx] << " ";
            }
            std::cout << "\n";
        }
    }

    // Print statistics for each channel
    for (int c = 0; c < 3; c++) {
        float sum = 0.0f;
        float min_val = 1.0f;
        float max_val = 0.0f;
        
        for (int i = 0; i < 32 * 32; i++) {
            int idx = (c * 32 * 32) + i;
            sum += image_data[idx];
            min_val = std::min(min_val, image_data[idx]);
            max_val = std::max(max_val, image_data[idx]);
        }
        
        float mean = sum / (32 * 32);
        std::cout << "\nChannel " << c << " statistics:";
        std::cout << "\n  Mean: " << mean;
        std::cout << "\n  Min:  " << min_val;
        std::cout << "\n  Max:  " << max_val << "\n";
    }
}

int main() {
    try {
        // Initialize dataset 
        DataSet dataset("../data");
        
        // Load the data and transfer to GPU
        dataset.load_data();
        dataset.to_gpu();
        
        // Allocate GPU memory for one image and label
        float* d_image;
        uint8_t* d_label;
        float* h_image = new float[3 * 32 * 32];
        uint8_t* h_label = new uint8_t[10];

        CHECK_CUDA_ERROR(cudaMalloc(&d_image, 3 * 32 * 32 * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_label, 10 * sizeof(uint8_t)));

        // Get first image and its label
        dataset.get_batch_data(d_image, d_label, 0, 1);
        
        // Copy data back to host for printing
        CHECK_CUDA_ERROR(cudaMemcpy(h_image, d_image, 3 * 32 * 32 * sizeof(float), 
                                  cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_label, d_label, 10 * sizeof(uint8_t), 
                                  cudaMemcpyDeviceToHost));
        
        // Print the image and label
        print_image(h_image, h_label, 10);
        
        // Cleanup
        delete[] h_image;
        delete[] h_label;
        cudaFree(d_image);
        cudaFree(d_label);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 