#pragma once
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"



static const int NUM_IMAGES_TOTAL = 50000;
static const int IMAGE_SIZE = 3 * 32 * 32;
static const int NUM_CLASSES = 10;


class DataSet {
private:
    std::string data_path_root; // folder path
    float* d_images;            // GPU memory for images
    uint8_t* d_labels;          // GPU memory for labels
    std::vector<float> h_images;     // CPU memory for images
    std::vector<uint8_t> h_labels; // CPU memory for labels

public:
    DataSet(const std::string& data_path_root);
    ~DataSet();
    
    void load_data();
    void to_gpu(); 
    
    void get_batch_data(float* d_batch_images, uint8_t* d_batch_labels, 
                       int batch_index, int batch_size);
    
    int get_num_batches(int batch_size) const {
        return NUM_IMAGES_TOTAL / batch_size;
    }
    const std::vector<float>& get_images() const { return h_images; }
    const std::vector<uint8_t>& get_labels() const { return h_labels; }
};