#include "include/dataset.cuh"
#include <fstream>
#include <stdexcept>
#include <iostream>

DataSet::DataSet(const std::string& path) : data_path_root(path) {
    d_images = nullptr;
    d_labels = nullptr;

    h_images.resize(NUM_IMAGES_TOTAL * IMAGE_SIZE);
    h_labels.resize(NUM_IMAGES_TOTAL * NUM_CLASSES);
}

DataSet::~DataSet() {
    if (d_images) cudaFree(d_images);
    if (d_labels) cudaFree(d_labels);
}

void DataSet::load_data() {
    for (int batch = 1; batch <= 5; batch++) {
        std::string data_path = data_path_root + "/data_batch_" + std::to_string(batch) + ".bin";
        std::ifstream file(data_path, std::ios::binary);
        
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open file " + data_path);
        }

        int offset = (batch - 1) * 10000;  // Each file has 10000 images

        // Read each image in the batch
        for (int i = 0; i < 10000; i++) {
            int img_idx = offset + i;
            
            // Read label (1 byte)
            uint8_t label;
            file.read(reinterpret_cast<char*>(&label), 1);

            // Convert to one-hot encoding
            for (int j = 0; j < NUM_CLASSES; j++) {
                h_labels[img_idx * NUM_CLASSES + j] = (j == label) ? 1 : 0;
            }

            // Read image data (3072 bytes = 32*32*3)
            std::vector<uint8_t> temp_buffer(IMAGE_SIZE);
            file.read(reinterpret_cast<char*>(temp_buffer.data()), IMAGE_SIZE);

            // Convert and normalize image data
            for (int j = 0; j < IMAGE_SIZE; j++) {
                float pixel_value = static_cast<float>(temp_buffer[j]) / 255.0f;
                h_images[img_idx * IMAGE_SIZE + j] = pixel_value;
            }
        }
        
        file.close();
    }

    std::cout << "Loaded " << NUM_IMAGES_TOTAL << " images successfully." << std::endl;
}

void DataSet::to_gpu() {
    if (d_images) {
        CHECK_CUDA_ERROR(cudaFree(d_images));
        d_images = nullptr;
    }
    if (d_labels) {
        CHECK_CUDA_ERROR(cudaFree(d_labels));
        d_labels = nullptr;
    }
    CHECK_CUDA_ERROR(cudaMalloc(&d_images, NUM_IMAGES_TOTAL * IMAGE_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, NUM_IMAGES_TOTAL * NUM_CLASSES * sizeof(uint8_t)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_images, h_images.data(), 
                         NUM_IMAGES_TOTAL * IMAGE_SIZE * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_labels.data(), 
                         NUM_IMAGES_TOTAL * NUM_CLASSES * sizeof(uint8_t), 
                         cudaMemcpyHostToDevice));
    
    std::vector<float>().swap(h_images);
    std::vector<uint8_t>().swap(h_labels);
}
void DataSet::get_batch_data(float* d_batch_images, uint8_t* d_batch_labels, 
                           int batch_index, int batch_size) {
    // Calculate starting position
    size_t image_offset = batch_index * batch_size * IMAGE_SIZE;
    size_t label_offset = batch_index * batch_size * NUM_CLASSES;

    // Validate batch index
    if (batch_index * batch_size >= NUM_IMAGES_TOTAL) {
        throw std::runtime_error("Batch index out of range");
    }

    // Calculate actual batch size (might be smaller for last batch)
    int remaining_images = NUM_IMAGES_TOTAL - batch_index * batch_size;
    int actual_batch_size = std::min(batch_size, remaining_images);

    // Copy batch data
    size_t image_copy_size = actual_batch_size * IMAGE_SIZE * sizeof(float);
    size_t label_copy_size = actual_batch_size * NUM_CLASSES * sizeof(uint8_t);

    CHECK_CUDA_ERROR(cudaMemcpy(d_batch_images, d_images + image_offset, 
                         image_copy_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_batch_labels, d_labels + label_offset, 
                         label_copy_size, cudaMemcpyDeviceToDevice));
}