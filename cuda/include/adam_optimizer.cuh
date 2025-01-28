#pragma once
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

class AdamOptimizer {
private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t;  // time step

    // Device pointers for momentum and velocity
    float *d_m;  // First moment
    float *d_v;  // Second moment
    size_t param_size;

    void allocate_memory(size_t size);
    void free_memory();

public:
    AdamOptimizer(float lr = 0.001f, float beta1 = 0.9f, 
                  float beta2 = 0.999f, float eps = 1e-8f);
    ~AdamOptimizer();

    void init(size_t num_params);
    void update(float* d_params, const float* d_gradients);
};