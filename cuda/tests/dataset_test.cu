#include "../include/dataset.cuh"
#include <iostream>
#include <iomanip>

void print_image(const float* image_data, const float* label_data, int num_classes) {
    // Print label
    std::cout << "Label one-hot encoding: ";
    for (int i = 0; i < num_classes; i++) {
        std::cout << label_data[i] << " ";
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
        
        // Load the data
        dataset.load_data();
        
        // Get first image and its label
        float* image = new float[3 * 32 * 32];
        uint8_t* label = new uint8_t[10];  

        dataset.get_batch_data(image, label, 0, 1);
        
        // Print the image and label
        print_image(image, label, 10);
        
        // Cleanup
        delete[] image;
        delete[] label;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 