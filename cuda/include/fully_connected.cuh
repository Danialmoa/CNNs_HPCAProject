#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "cuda_utils.cuh"
#include "adam_optimizer.cuh"
#include <cstdint>

class FullyConnectedLayer {
private:
    int in_features, num_classes;
    float learning_rate;
    
    // Device pointers
    float *d_weights, *d_biases;
    float *d_input_cache;    // Store input for backward pass
    float *d_output_cache;   // Store softmax output for backward pass
   
    //Adam Optimizer
   AdamOptimizer weights_optimizer;
    AdamOptimizer bias_optimizer;

    // Helper functions
    void allocate_memory(int batch_size);
    void free_memory();

public:
    FullyConnectedLayer(int in_features, int num_classes, float learning_rate);
    ~FullyConnectedLayer();

    void forward(const float* d_input, float* d_output, int batch_size);
    void backward(const uint8_t* d_labels, float* d_grad_input, int batch_size);
    float compute_loss(const uint8_t* d_labels, int batch_size);
};