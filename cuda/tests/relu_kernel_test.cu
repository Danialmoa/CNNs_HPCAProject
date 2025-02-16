#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// ReLU kernel with in-place operation
__global__ void relu_kernel(
    float* data,
    int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

// CPU reference implementation
void cpu_relu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = data[i] > 0 ? data[i] : 0;
    }
}

int main() {
    // Test parameters
    const int size = 32;  // Total elements

    // Create test data
    std::vector<float> h_data(size);
    for (int i = 0; i < size; i++) {
        h_data[i] = i - size/2;  // Will generate both positive and negative numbers
    }
    std::vector<float> h_data_cpu = h_data;  // Copy for CPU computation

    // Print input data
    std::cout << "Input data:\n";
    for (int i = 0; i < size; i++) {
        std::cout << h_data[i] << " ";
        if ((i + 1) % 8 == 0) std::cout << "\n";
    }
    std::cout << "\n";

    // Allocate device memory
    float *d_data;
    cudaMalloc(&d_data, size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;

    relu_kernel<<<blocks, threads_per_block>>>(d_data, size);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    cpu_relu(h_data_cpu.data(), size);

    // Print and compare results
    std::cout << "\nGPU Output:\n";
    for (int i = 0; i < size; i++) {
        std::cout << h_data[i] << " ";
        if ((i + 1) % 8 == 0) std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "\nCPU Output:\n";
    for (int i = 0; i < size; i++) {
        std::cout << h_data_cpu[i] << " ";
        if ((i + 1) % 8 == 0) std::cout << "\n";
    }
    std::cout << "\n";

    // Compare results
    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = std::abs(h_data_cpu[i] - h_data[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-5) {
            std::cout << "Mismatch at position " << i << ": CPU = " << h_data_cpu[i] 
                      << ", GPU = " << h_data[i] << "\n";
        }
    }
    std::cout << "\nMaximum difference between CPU and GPU: " << max_diff << std::endl;

    // Cleanup
    cudaFree(d_data);

    return 0;
} 