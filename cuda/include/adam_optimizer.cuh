#pragma once
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

class AdamOptimizer {
private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    
    int size;
    float* d_m;  // First moment
    float* d_v;  // Second moment
    int t;       // Timestep

public:
    AdamOptimizer(int param_size = 0, float lr = 0.001f, 
                  float b1 = 0.9f, float b2 = 0.999f, 
                  float eps = 1e-8f);
    ~AdamOptimizer();
    
    void update(float* params, const float* gradients);
};