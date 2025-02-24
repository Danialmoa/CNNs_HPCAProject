#include "include/fully_connected.cuh"
#include "include/adam_optimizer.cuh"
#include <random>
#include <cmath>
#include <iostream>

// CUDA kernels
__global__ void fc_forward_kernel(
    const float* input, const float* weights, const float* biases,
    float* output, int batch_size, int in_features, int num_classes) {
    
    int b = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || c >= num_classes) return;
    
    float sum = biases[c];
    for (int i = 0; i < in_features; i++) {
        sum += input[b * in_features + i] * weights[c * in_features + i];
    }
    
    output[b * num_classes + c] = sum;
}

__global__ void softmax_kernel(
    float* input, float* output,
    int batch_size, int num_classes) {
    
    int b = blockIdx.x;
    if (b >= batch_size) return;
    
    // Find max value for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < num_classes; i++) {
        max_val = fmaxf(max_val, input[b * num_classes + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    __shared__ float exp_values[10];

    for (int i = 0; i < num_classes; i++) {
        exp_values[i] = expf(input[b * num_classes + i] - max_val);
        sum += exp_values[i];
    }
    
    // Normalize
    for (int i = 0; i < num_classes; i++) {
        output[b * num_classes + i] = exp_values[i] / (sum + 1e-7f);
    }
}

__global__ void cross_entropy_loss_kernel(
    const float* softmax_output, const uint8_t* labels,
    float* loss, int batch_size, int num_classes) {
    
    int b = blockIdx.x;
    if (b >= batch_size) return;
    
    float batch_loss = 0.0f;
    
    // Find the true class and compute loss
    for (int i = 0; i < num_classes; i++) {
        if (labels[b * num_classes + i] == 1) {
            // Add small epsilon for numerical stability
            float pred = fmaxf(softmax_output[b * num_classes + i], 1e-10f);
            batch_loss = -logf(pred);
            
            break;
        }
    }
    
    // Accumulate loss
    atomicAdd(loss, batch_loss);
}

__global__ void backward_kernel(
    const float* softmax_output, const uint8_t* labels,
    const float* input_cache, const float* weights,
    float* grad_input, float* grad_weights, float* grad_biases,
    int batch_size, int in_features, int num_classes) {
    
    int b = blockIdx.x;
    int c = blockIdx.y;
    
    if (b >= batch_size || c >= num_classes) return;
    
    int idx = b * num_classes + c;
    float grad = (softmax_output[idx] - labels[idx]) / batch_size;

    const float CLIP_VALUE = 1.0f;
    grad = fmaxf(fminf(grad, CLIP_VALUE), -CLIP_VALUE);
    
    // Gradient for biases
    atomicAdd(&grad_biases[c], grad);
    
    // Gradient for weights and inputs
    const float l2_reg = 0.0001f;  // L2 regularization strength
    for (int i = 0; i < in_features; i++) {
        float weight_grad = input_cache[b * in_features + i] * grad + l2_reg * weights[c * in_features + i];
        atomicAdd(&grad_weights[c * in_features + i], weight_grad);
        atomicAdd(&grad_input[b * in_features + i], weights[c * in_features + i] * grad);
    }
}

FullyConnectedLayer::FullyConnectedLayer(int in_features, int num_classes, float learning_rate)
    : in_features(in_features), 
    num_classes(num_classes), 
    learning_rate(learning_rate),
    weights_optimizer(num_classes * in_features, learning_rate),
    bias_optimizer(num_classes, learning_rate),
    current_batch_size(0),
    d_input_cache(nullptr),
    d_output_cache(nullptr),
    d_weights(nullptr),
    d_biases(nullptr)
    {

    // Initialize weights and biases on CPU
    std::vector<float> h_weights(num_classes * in_features);
    std::vector<float> h_biases(num_classes);
    
    // Xavier initialization
    float std_dev = sqrt(2.0f / in_features);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0f, std_dev);
    
    for (auto& w : h_weights) {
        w = distribution(gen);
    }
    std::fill(h_biases.begin(), h_biases.end(), 0.01f);

    // Allocate and copy to GPU
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights, h_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_biases, h_biases.size() * sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, h_weights.data(), 
                               h_weights.size() * sizeof(float), 
                               cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_biases, h_biases.data(), 
                               h_biases.size() * sizeof(float), 
                               cudaMemcpyHostToDevice));
    
}

FullyConnectedLayer::~FullyConnectedLayer() {
    free_memory();
    if (d_weights) {
        CHECK_CUDA_ERROR(cudaFree(d_weights));
        d_weights = nullptr;
    }
    if (d_biases) {
        CHECK_CUDA_ERROR(cudaFree(d_biases));
        d_biases = nullptr;
    }
}

void FullyConnectedLayer::allocate_memory(int batch_size) {
     if (current_batch_size != batch_size) {
            // Free existing memory
            if (d_input_cache) cudaFree(d_input_cache);
            if (d_output_cache) cudaFree(d_output_cache);

            // Allocate new memory
            CHECK_CUDA_ERROR(cudaMalloc(&d_input_cache, 
                                      batch_size * in_features * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_output_cache, 
                                      batch_size * num_classes * sizeof(float)));
            
        current_batch_size = batch_size;
    }
}

void FullyConnectedLayer::free_memory() {
    if (d_input_cache) cudaFree(d_input_cache);
    if (d_output_cache) cudaFree(d_output_cache);
}

void FullyConnectedLayer::forward(const float* d_input, float* d_output, int batch_size) {

    //Debug
    float* h_input = new float[batch_size * in_features];
    cudaMemcpy(h_input, d_input, batch_size * in_features * sizeof(float), cudaMemcpyDeviceToHost);
    float input_sum = 0;
    for(int i = 0; i < 10; i++) {
        input_sum += h_input[i];
    }
    std::cout << "Input first 10 values sum: " << input_sum << std::endl;
    delete[] h_input;

    // Allocate memory for caches
    allocate_memory(batch_size);
    
    // Cache input for backward pass
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_cache, d_input, 
                               batch_size * in_features * sizeof(float), 
                               cudaMemcpyDeviceToDevice));
    
    // Fully connected layer
    dim3 fc_block(256);
    dim3 fc_grid(batch_size, (num_classes + fc_block.x - 1) / fc_block.x);
    
    fc_forward_kernel<<<fc_grid, fc_block>>>(
        d_input, d_weights, d_biases,
        d_output_cache, batch_size, in_features, num_classes
    );
    CHECK_LAST_CUDA_ERROR();


    // Debug: Check FC output
    float* h_fc_output = new float[batch_size * num_classes];
    cudaMemcpy(h_fc_output, d_output_cache, batch_size * num_classes * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "FC output first sample: ";
    for(int i = 0; i < num_classes; i++) {
        std::cout << h_fc_output[i] << " ";
    }
    std::cout << std::endl;
    delete[] h_fc_output;
    
    // Softmax
    softmax_kernel<<<batch_size, 1>>>(
        d_output_cache, d_output,
        batch_size, num_classes
    );
    CHECK_LAST_CUDA_ERROR();

    // Debug: Check softmax output
    float* h_softmax = new float[batch_size * num_classes];
    cudaMemcpy(h_softmax, d_output, batch_size * num_classes * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Softmax output first sample: ";
    for(int i = 0; i < num_classes; i++) {
        std::cout << h_softmax[i] << " ";
    }
    std::cout << std::endl;
    delete[] h_softmax;
    
    // Cache softmax output
    CHECK_CUDA_ERROR(cudaMemcpy(d_output_cache, d_output, 
                               batch_size * num_classes * sizeof(float), 
                               cudaMemcpyDeviceToDevice));
}

void FullyConnectedLayer::backward(const uint8_t* d_labels, float* d_grad_input, int batch_size) {
    size_t input_grad_size = batch_size * in_features;
    size_t weight_size = num_classes * in_features;
    size_t bias_size = num_classes;

    // Allocate memory for gradients
    float *d_grad_weights, *d_grad_biases;
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_weights, weight_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_biases, bias_size * sizeof(float)));
    
    // Zero out gradients
    CHECK_CUDA_ERROR(cudaMemset(d_grad_weights, 0, weight_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_biases, 0, bias_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_input, 0, input_grad_size * sizeof(float)));

    // Compute gradients
    dim3 grid(batch_size, num_classes);
    backward_kernel<<<grid, 1>>>(
        d_output_cache, d_labels,
        d_input_cache, d_weights,
        d_grad_input, d_grad_weights, d_grad_biases,
        batch_size, in_features, num_classes
    );
    CHECK_LAST_CUDA_ERROR();
    
    // Update weights and biases using Adam optimizer
    weights_optimizer.update(d_weights, d_grad_weights);
    bias_optimizer.update(d_biases, d_grad_biases);
    
    // Free temporary memory
    cudaFree(d_grad_weights);
    cudaFree(d_grad_biases);
}

float FullyConnectedLayer::compute_loss(const uint8_t* d_labels, int batch_size) {
    float* d_loss;
    float h_loss;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_loss, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_loss, 0, sizeof(float)));
    
    cross_entropy_loss_kernel<<<batch_size, 1>>>(
        d_output_cache, d_labels,
        d_loss, batch_size, num_classes
    );
    CHECK_LAST_CUDA_ERROR();
    
    CHECK_CUDA_ERROR(cudaMemcpy(&h_loss, d_loss, sizeof(float), 
                               cudaMemcpyDeviceToHost));
    
    cudaFree(d_loss);
    
    return h_loss / batch_size;
}