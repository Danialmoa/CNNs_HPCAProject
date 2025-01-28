#include "include/dataset.cuh"
#include <fstream>
#include <stdexcept>
#include <iostream>

DataSet::DataSet(const std::string& data_path_root) 
    : data_path_root(data_path_root), d_images(nullptr), d_labels(nullptr) {
    
    h_labels.resize(NUM_BATCHES * NUM_IMAGES_PER_BATCH, std::vector<uint8_t>(NUM_CLASSES, 0));
    h_images.resize(NUM_BATCHES * NUM_IMAGES_PER_BATCH * IMAGE_SIZE, 0.0f);
}

DataSet::~DataSet() {
    if (d_images) {
        CHECK_CUDA_ERROR(cudaFree(d_images));
    }
    if (d_labels) {
        CHECK_CUDA_ERROR(cudaFree(d_labels));
    }
}

void DataSet::load_data() {
    const float CIFAR_MEANS[3] = {0.4914f, 0.4822f, 0.4465f};
    const float CIFAR_STDS[3] = {0.2470f, 0.2435f, 0.2616f};

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NUM_BATCHES; i++) {
        std::string data_path = data_path_root + "/data_batch_" + std::to_string(i + 1) + ".bin";
        std::ifstream file(data_path, std::ios::binary);
        
        if (!file.is_open()) {
            #pragma omp critical
            {
                throw std::runtime_error("Error: Could not open file " + data_path);
            }
            continue;
        }

        for (int j = 0; j < NUM_IMAGES_PER_BATCH; ++j) {
            int image_offset = (i * NUM_IMAGES_PER_BATCH + j) * IMAGE_SIZE;

            // Read label
            uint8_t label;
            file.read(reinterpret_cast<char*>(&label), 1);
            h_labels[i * NUM_IMAGES_PER_BATCH + j][label] = 1;

            // Read and normalize image data
            for (int k = 0; k < 3; ++k) {
                for (int h = 0; h < 32; ++h) {
                    for (int w = 0; w < 32; ++w) {
                        uint8_t pixel;
                        file.read(reinterpret_cast<char*>(&pixel), 1);
                        int idx = image_offset + k * 32 * 32 + h * 32 + w;
                        h_images[idx] = (pixel / 255.0f - CIFAR_MEANS[k]) / CIFAR_STDS[k];
                    }
                }
            }
        }
        file.close();
    }
}

void DataSet::to_gpu() {
    size_t images_size = NUM_BATCHES * NUM_IMAGES_PER_BATCH * IMAGE_SIZE * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_images, images_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_images, h_images.data(), images_size, cudaMemcpyHostToDevice));

    // Allocate GPU memory for labels
    size_t labels_size = NUM_BATCHES * NUM_IMAGES_PER_BATCH * NUM_CLASSES * sizeof(uint8_t);
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, labels_size));
    
    // Create contiguous array for labels
    std::vector<uint8_t> h_labels_contiguous(NUM_BATCHES * NUM_IMAGES_PER_BATCH * NUM_CLASSES);
    for (int i = 0; i < h_labels.size(); i++) {
        std::copy(h_labels[i].begin(), h_labels[i].end(), 
                 h_labels_contiguous.begin() + i * NUM_CLASSES);
    }
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_labels_contiguous.data(), 
                               labels_size, cudaMemcpyHostToDevice));

    // Free CPU memory after transfer
    std::vector<float>().swap(h_images);
    std::vector<std::vector<uint8_t>>().swap(h_labels);
}

void DataSet::get_batch_data(float* d_batch_images, uint8_t* d_batch_labels, 
                            int batch_index, int batch_size) {

    size_t image_offset = batch_index * batch_size * IMAGE_SIZE;
    size_t label_offset = batch_index * batch_size * NUM_CLASSES;
    size_t elements_to_copy = batch_size * IMAGE_SIZE;

    size_t byte_offset = image_offset * sizeof(float);
    size_t bytes_to_copy = elements_to_copy * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_batch_images, 
                               reinterpret_cast<float*>(reinterpret_cast<char*>(d_images) + byte_offset),
                               bytes_to_copy, 
                               cudaMemcpyDeviceToDevice));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_batch_labels, 
                               d_labels + label_offset, 
                               batch_size * NUM_CLASSES * sizeof(uint8_t), 
                               cudaMemcpyDeviceToDevice));
}