#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Convolution forward kernel
__global__ void conv_forward_kernel(
    const float* input,           // [batch, in_channels, height, width]
    const float* weights,         // [out_channels, in_channels, kernel, kernel]
    const float* biases,          // [out_channels]
    float* output,                // [batch, out_channels, out_height, out_width]
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int out_height,
    int out_width
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_ch = blockIdx.z % out_channels;
    const int batch = blockIdx.z / out_channels;
    
    if (x >= out_width || y >= out_height || out_ch >= out_channels || batch >= batch_size) return;
    
    float sum = biases[out_ch];
    
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = y * stride - padding + kh;
                int w_in = x * stride - padding + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = ((batch * in_channels + in_ch) * height + h_in) * width + w_in;
                    int weight_idx = ((out_ch * in_channels + in_ch) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    output[((batch * out_channels + out_ch) * out_height + y) * out_width + x] = sum;
}

// CPU reference implementation
void cpu_convolution(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size,
    int stride, int padding, int out_height, int out_width) {
    
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float sum = biases[oc];
                    
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int h_in = oh * stride - padding + kh;
                                int w_in = ow * stride - padding + kw;
                                
                                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                                    int input_idx = ((b * in_channels + ic) * height + h_in) * width + w_in;
                                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                    sum += input[input_idx] * weights[weight_idx];
                                }
                            }
                        }
                    }
                    
                    output[((b * out_channels + oc) * out_height + oh) * out_width + ow] = sum;
                }
            }
        }
    }
}

int main() {
    // Test parameters
    const int batch_size = 1;
    const int in_channels = 1;
    const int out_channels = 1;
    const int height = 4;
    const int width = 4;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 1;
    
    // Calculate output dimensions
    const int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (width + 2 * padding - kernel_size) / stride + 1;

    // Create test data
    std::vector<float> h_input(batch_size * in_channels * height * width, 1.0f);
    std::vector<float> h_weights(out_channels * in_channels * kernel_size * kernel_size, 1.0f);
    std::vector<float> h_biases(out_channels, 0.0f);
    std::vector<float> h_output(batch_size * out_channels * out_height * out_width);
    std::vector<float> h_output_cpu(h_output.size());

    // Print input data
    std::cout << "Input data (4x4):\n";
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            std::cout << h_input[h * width + w] << " ";
        }
        std::cout << "\n";
    }

    // Allocate device memory
    float *d_input, *d_weights, *d_biases, *d_output;
    cudaMalloc(&d_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_weights, h_weights.size() * sizeof(float));
    cudaMalloc(&d_biases, h_biases.size() * sizeof(float));
    cudaMalloc(&d_output, h_output.size() * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, h_biases.data(), h_biases.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (out_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (out_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
        out_channels * batch_size
    );

    conv_forward_kernel<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weights, d_biases, d_output,
        batch_size, in_channels, out_channels,
        height, width, kernel_size, stride, padding,
        out_height, out_width
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    cpu_convolution(h_input.data(), h_weights.data(), h_biases.data(), h_output_cpu.data(),
                   batch_size, in_channels, out_channels, height, width,
                   kernel_size, stride, padding, out_height, out_width);

    // Print and compare results
    std::cout << "\nGPU Output:\n";
    for (int h = 0; h < out_height; h++) {
        for (int w = 0; w < out_width; w++) {
            std::cout << h_output[h * out_width + w] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nCPU Output:\n";
    for (int h = 0; h < out_height; h++) {
        for (int w = 0; w < out_width; w++) {
            std::cout << h_output_cpu[h * out_width + w] << " ";
        }
        std::cout << "\n";
    }

    // Compare results
    float max_diff = 0.0f;
    for (size_t i = 0; i < h_output.size(); i++) {
        float diff = std::abs(h_output_cpu[i] - h_output[i]);
        max_diff = std::max(max_diff, diff);
    }
    std::cout << "\nMaximum difference between CPU and GPU: " << max_diff << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_output);

    return 0;
} 