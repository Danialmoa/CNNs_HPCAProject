#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <array>
#include <limits>
#include <omp.h>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>


class DataSet {
    private:
        std::string data_path_root; // folder path
        std::vector<std::vector<uint8_t> > labels; // 50000x10
        std::vector<float> images; // 50000x3x32x32
        static const int NUM_IMAGES_PER_BATCH = 10000;
        static const int NUM_BATCHES = 5;
        static const int IMAGE_SIZE = 3 * 32 * 32;

    public:
        const auto& getImages() const { return images; }
        const auto& getLabels() const { return labels; }

        DataSet(const std::string& data_path_root) 
            : data_path_root(data_path_root) {
                labels.resize(NUM_BATCHES * NUM_IMAGES_PER_BATCH, std::vector<uint8_t>(10, 0));
                images.resize(NUM_BATCHES * NUM_IMAGES_PER_BATCH * IMAGE_SIZE, 0.0f);
            }

        void load_data() {
            #pragma omp parallel for schedule(static) num_threads(5)
            for (int i = 0; i < NUM_BATCHES; i++) {
                std::string data_path = data_path_root + "/data_batch_" + std::to_string(i + 1) + ".bin";
                std::ifstream file(data_path, std::ios::binary);
                if (!file.is_open()) {
                    #pragma omp critical
                    throw std::runtime_error("Error: Could not open file " + data_path);
                    continue;
                }
                for (int j = 0; j < NUM_IMAGES_PER_BATCH; ++j) {
                    int image_offset = (i * NUM_IMAGES_PER_BATCH + j) * IMAGE_SIZE;

                    // Read label
                    uint8_t label;
                    file.read(reinterpret_cast<char*>(&label), 1);
                    labels[i * NUM_IMAGES_PER_BATCH + j][label] = 1;

                    // Read image data
                    // store as 3x32x32
                    for (int k = 0; k < 3; ++k) {
                        for (int h = 0; h < 32; ++h) {
                            for (int w = 0; w < 32; ++w) {
                                uint8_t pixel;
                                file.read(reinterpret_cast<char*>(&pixel), 1);
                                images[image_offset + k * 32 * 32 + h * 32 + w] = pixel / 255.0f;
                                const float CIFAR_MEANS[3] = {0.4914f, 0.4822f, 0.4465f};
                                const float CIFAR_STDS[3] = {0.2470f, 0.2435f, 0.2616f};
                                images[image_offset + k * 32 * 32 + h * 32 + w] = 
                                    (pixel / 255.0f - CIFAR_MEANS[k]) / CIFAR_STDS[k];
                            }
                        }
                    }
                }
                file.close();
            }
        }
        
        std::vector<float> getBatchImages(int batch_index, int batch_size) {
            int start_idx = batch_index * batch_size * IMAGE_SIZE;
            int end_idx = std::min(start_idx + batch_size * IMAGE_SIZE, static_cast<int>(images.size()));
            return std::vector<float>(
                images.begin() + start_idx,
                images.begin() + end_idx
            );
        }
        std::vector<std::vector<uint8_t>> getBatchLabels(int batch_idx, int batch_size) const {
            int start_idx = batch_idx * batch_size;
            int end_idx = std::min(start_idx + batch_size, static_cast<int>(labels.size()));
            return std::vector<std::vector<uint8_t>>(
                labels.begin() + start_idx,
                labels.begin() + end_idx
            );
        }

        int getNumBatches(int batch_size) const {
            return (NUM_BATCHES * NUM_IMAGES_PER_BATCH) / batch_size; // ceil division
        }
        ~DataSet() {
            clear();
        }
        void clear() {
            images.clear();
            labels.clear();
        }
    };


class AdamOptimizer {
private:
    float beta1;
    float beta2;
    float epsilon;
    float learning_rate;
    int t;

    std::vector<float> m;
    std::vector<float> v;
    
public:
    AdamOptimizer(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
        : learning_rate(lr), beta1(beta1), beta2(beta2), epsilon(eps), t(0) {
        }
    
    void init(size_t num_params) {
        m.resize(num_params, 0.0f);
        v.resize(num_params, 0.0f);
    }
    
    void update(std::vector<float>& params, const std::vector<float>& gradients) {
        if (m.empty()) {
            init(params.size());
        }
        t++;
        float alpha_t = learning_rate * std::sqrt(1.0f - std::pow(beta2, t)) / 
                       (1.0f - std::pow(beta1, t));
        #pragma omp parallel
        {   
            #pragma omp for
            for (size_t i = 0; i < params.size(); i++) {
                float scaled_grad = gradients[i];
                m[i] = beta1 * m[i] + (1.0f - beta1) * scaled_grad;
                v[i] = beta2 * v[i] + (1.0f - beta2) * scaled_grad * scaled_grad;
                
                float update = alpha_t * m[i] / (std::sqrt(v[i]) + epsilon);
                update = std::max(std::min(update, 1.0f), -1.0f);

                    params[i] -= update;
            }
        }
    }
};

class ConvBlock {
    // Convolutional Layer + ReLU + MaxPooling
    private:
        int in_channels, out_channels, kernel_size, stride, padding;
        int pool_size, pool_stride;
        std::vector<float> weights; 
        std::vector<float> biases;
        std::vector<float> cache;
        std::vector<float> conv_output_cache;
        std::vector<float> relu_output_cache;
        std::vector<int> pool_indices;
        float learning_rate;
        AdamOptimizer weights_optimizer;
        AdamOptimizer bias_optimizer;

        int input_height, input_width;
        int output_height, output_width;

        // Helper function to get weight index
        inline int helper_weight_idx(int oc, int ic, int kh, int kw) const {
            return ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
        }
        
        // Helper function to get input/output index
        inline int helper_feat_idx(int n, int c, int h, int w, int channels, int height, int width) const {
            return ((n * channels + c) * height + h) * width + w;
        }

    public:
        ConvBlock(int in_channels, int out_channels, int kernel_size, int stride, int padding, 
            int pool_size, int pool_stride, float learning_rate)
            : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size),
              stride(stride), padding(padding), pool_size(pool_size), pool_stride(pool_stride), learning_rate(learning_rate)
              {
            weights_optimizer = AdamOptimizer(learning_rate);
            bias_optimizer = AdamOptimizer(learning_rate);

            weights.resize(out_channels * in_channels * kernel_size * kernel_size);
            biases.resize(out_channels);

            float std_dev = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> distribution(0.0f, std_dev);

            for (auto& w : weights) {
                w = distribution(gen);
            }
            std::fill(biases.begin(), biases.end(), 0.01f);
        }

        std::vector<float> forward(const std::vector<float>& input, int batch_size, int height, int width) {
            // Input is in NCHW format (batch_size x channels x height x width)
            // Store dimensions for backward pass
            input_height = height;
            input_width = width;
            cache = input;

            int conv_height = (height + 2 * padding - kernel_size) / stride + 1;
            int conv_width = (width + 2 * padding - kernel_size) / stride + 1;
            int pool_output_height = (conv_height - pool_size) / pool_stride + 1;
            int pool_output_width = (conv_width - pool_size) / pool_stride + 1;
            
            // Output is in NCHW format (batch_size x channels x height x width)
            conv_output_cache.resize(batch_size * out_channels * conv_height * conv_width);
            relu_output_cache.resize(batch_size * out_channels * conv_height * conv_width);
            std::vector<float> output(batch_size * out_channels * pool_output_height * pool_output_width, 0.0f);
            pool_indices.resize(output.size());

            #pragma omp parallel
            {
                #pragma omp for collapse(4)
                for (int b = 0; b < batch_size; ++b) {
                    for (int oc = 0; oc < out_channels; ++oc) {
                        for (int ph = 0; ph < pool_output_height; ++ph) {
                            for (int pw = 0; pw < pool_output_width; ++pw) {
                            float max_pool_val = -std::numeric_limits<float>::infinity();
                            int max_pool_idx = -1;

                            for (int pool_i = 0; pool_i < pool_size; ++pool_i) {
                            for (int pool_j = 0; pool_j < pool_size; ++pool_j) {
                                int conv_h = ph * pool_stride + pool_i;
                                int conv_w = pw * pool_stride + pool_j;
                                
                                float conv_result = biases[oc];
                                for (int ic = 0; ic < in_channels; ++ic) {
                                    for (int kh = 0; kh < kernel_size; ++kh) {
                                        for (int kw = 0; kw < kernel_size; ++kw) {
                                            int ih = conv_h * stride - padding + kh;
                                            int iw = conv_w * stride - padding + kw;

                                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                                int input_idx = helper_feat_idx(b, ic, ih, iw, in_channels, height, width);
                                                int weight_idx = helper_weight_idx(oc, ic, kh, kw);
                                                conv_result += input[input_idx] * weights[weight_idx];
                                            }
                                        }
                                    }
                                }

                            // Store conv result
                            int conv_idx = helper_feat_idx(b, oc, conv_h, conv_w, out_channels, conv_height, conv_width);
                            conv_output_cache[conv_idx] = conv_result;

                            // ReLU
                            float relu_result = std::max(0.0f, conv_result);
                            relu_output_cache[conv_idx] = relu_result;

                            // Update max pooling
                            if (relu_result > max_pool_val) {
                                max_pool_val = relu_result;
                                max_pool_idx = conv_idx;
                            }
                            }
                        }
                        // Store pooling result
                        int output_idx = helper_feat_idx(b, oc, ph, pw, out_channels, 
                                            pool_output_height, pool_output_width);
                        output[output_idx] = max_pool_val;
                        pool_indices[output_idx] = max_pool_idx;
                            }
                        }
                    }
                }
            }
            return output;
        }

        std::vector<float> backward(const std::vector<float>& grad_output) {
        std::vector<float> input_gradients(cache.size(), 0.0f);
        std::vector<float> weight_gradients(weights.size(), 0.0f);
        std::vector<float> bias_gradients(biases.size(), 0.0f);

        int batch_size = grad_output.size() / (out_channels * ((input_height + 2 * padding - kernel_size) / stride + 1) * 
                                             ((input_width + 2 * padding - kernel_size) / stride + 1));
        
        // Calculate dimensions
        int conv_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        int conv_width = (input_width + 2 * padding - kernel_size) / stride + 1;
        int pool_output_height = (conv_height - pool_size) / pool_stride + 1;
        int pool_output_width = (conv_width - pool_size) / pool_stride + 1;


        #pragma omp parallel
        {
            #pragma omp for collapse(4)
            for (int b = 0; b < batch_size; ++b) {
                for (int oc = 0; oc < out_channels; ++oc) {
                for (int h = 0; h < conv_height; ++h) {
                    for (int w = 0; w < conv_width; ++w) {
                        int conv_idx = helper_feat_idx(b, oc, h, w, out_channels, conv_height, conv_width);
                        
                        // Get pooling gradient
                        float pool_grad = 0.0f;
                        int pool_h = h / pool_stride;
                        int pool_w = w / pool_stride;
                        
                        if (h % pool_stride == 0 && w % pool_stride == 0 && 
                            pool_h < pool_output_height && pool_w < pool_output_width) {
                            
                            int pool_idx = helper_feat_idx(b, oc, pool_h, pool_w, out_channels, 
                                                  pool_output_height, pool_output_width);
                            
                            if (pool_indices[pool_idx] == conv_idx) {
                                pool_grad = grad_output[pool_idx];
                            }
                        }
                        // ReLU gradient
                        float relu_grad = (relu_output_cache[conv_idx] > 0) ? pool_grad : 0.0f;

                        // Skip if no gradient
                        if (relu_grad == 0.0f) continue;

                        // Convolution gradient
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int ih = h * stride - padding + kh;
                                    int iw = w * stride - padding + kw;

                                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                        int input_idx = helper_feat_idx(b, ic, ih, iw, in_channels, input_height, input_width);
                                        int weight_idx = helper_weight_idx(oc, ic, kh, kw);

                                        #pragma omp atomic
                                        input_gradients[input_idx] += weights[weight_idx] * relu_grad;

                                        #pragma omp atomic
                                        weight_gradients[weight_idx] += cache[input_idx] * relu_grad;
                                    }
                                }
                            }

                            #pragma omp atomic
                                bias_gradients[oc] += relu_grad;
                            }
                        }
                    }
                }
            }
        }
        weights_optimizer.update(weights, weight_gradients);
        bias_optimizer.update(biases, bias_gradients);

        return input_gradients;
    }
};



class FullyConnectedLayer {
    // Fully Connected Layer & SoftMax & Cross Entropy Loss
    private:
        int in_channels, num_classes;
        std::vector<float> weights;
        std::vector<float> biases;
        std::vector<float> input_cache;
        std::vector<float> output_cache;
        float learning_rate;
        AdamOptimizer weights_optimizer;
        AdamOptimizer bias_optimizer;

        inline int get_input_size() const {
            return in_channels;
        }

    public:
        FullyConnectedLayer(int in_channels, int num_classes, float learning_rate) 
            : in_channels(in_channels), num_classes(num_classes), learning_rate(learning_rate) {
            
            weights_optimizer = AdamOptimizer(learning_rate);
            bias_optimizer = AdamOptimizer(learning_rate);

            weights.resize(num_classes * in_channels);
            biases.resize(num_classes);

            float std_dev = std::sqrt(2.0f / in_channels);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> distribution(0.0f, std_dev);

            for (auto& w : weights) {
                w = distribution(gen);
            }
            std::fill(biases.begin(), biases.end(), 0.01f);
        }

        std::vector<float> forward(const std::vector<float>& input, int batch_size) {
            input_cache.resize(batch_size * in_channels);
            output_cache.resize(batch_size * num_classes);
            std::vector<float> output(batch_size * num_classes);

            input_cache = input;
            #pragma omp parallel
            {
                #pragma omp for
                for (int b = 0; b < batch_size; ++b) {
                    // Fully connected forward
                    std::vector<float> fc_output(num_classes);
                    for (int i = 0; i < num_classes; ++i) {
                        float sum = biases[i];
                        #pragma omp simd reduction(+:sum)
                        for (int j = 0; j < in_channels; ++j) {
                            sum += input[b * in_channels + j] * weights[i * in_channels + j];
                        }
                        fc_output[i] = sum;
                    }

                    // Softmax forward (with numerical stability)
                    float max_val = -std::numeric_limits<float>::infinity();
                    #pragma omp simd reduction(max:max_val)
                    for (int j = 0; j < num_classes; ++j) {
                        max_val = std::max(max_val, fc_output[j]);
                    }

                    float sum = 0.0f;
                    #pragma omp simd reduction(+:sum)
                    for (int j = 0; j < num_classes; ++j) {
                        output[b * num_classes + j] = std::exp(fc_output[j] - max_val);
                        sum += output[b * num_classes + j];
                    }

                    const float inv_sum = 1.0f / sum;
                    #pragma omp simd
                    for (int j = 0; j < num_classes; ++j) {
                        const int idx = b * num_classes + j;
                        output[idx] *= inv_sum;
                        output_cache[idx] = output[idx];
                    }
                }
            }
            return output;
        }

        std::vector<float> backward(const std::vector<std::vector<uint8_t>>& labels, int batch_size) {

            std::vector<float> grad_input(batch_size * in_channels, 0.0f);
            std::vector<float> weight_gradients(weights.size(), 0.0f);
            std::vector<float> bias_gradients(biases.size(), 0.0f);

            #pragma omp parallel
            {
                #pragma omp for collapse(2)
                for (int b = 0; b < batch_size; ++b) {
                    for (int i = 0; i < num_classes; ++i) {
                        const float grad = (output_cache[b * num_classes + i] - labels[b][i]) / batch_size;

                    // Accumulate bias gradients
                    #pragma omp atomic
                    bias_gradients[i] += grad;

                    // Compute weight gradients
                    for (int j = 0; j < in_channels; ++j) {
                        const int weight_idx = i * in_channels + j;
                        const int input_idx = b * in_channels + j;
                        
                        #pragma omp atomic
                        weight_gradients[weight_idx] += grad * input_cache[input_idx];
                        
                        #pragma omp atomic
                        grad_input[input_idx] += grad * weights[weight_idx];
                    }   
                    }
                }
            }
            weights_optimizer.update(weights, weight_gradients);
            bias_optimizer.update(biases, bias_gradients);

            return grad_input;
        }
    
        float compute_loss(const std::vector<std::vector<uint8_t>>& labels) const {
            float loss = 0.0f;
            const float epsilon = 1e-10f;
            const int batch_size = labels.size();

            #pragma omp parallel
            {
                #pragma omp for collapse(2) reduction(+:loss)
                for (int b = 0; b < batch_size; ++b) {
                    for (int j = 0; j < num_classes; ++j) {
                        if (labels[b][j] == 1) {
                            const int idx = b * num_classes + j;
                            loss += -std::log(std::max(output_cache[idx], epsilon));
                        }
                    }
                }
            }
            return loss / batch_size;
        }
};





class Train {
    private:
        DataSet data_set;
        ConvBlock conv_layer_1;
        ConvBlock conv_layer_2;
        ConvBlock conv_layer_3;
        FullyConnectedLayer fully_connected_layer;
        float learning_rate;

        float calculate_accuracy(const std::vector<float>& predictions, 
                               const std::vector<std::vector<uint8_t>>& labels,
                               int batch_size) {
            int correct = 0;
            
            #pragma omp parallel
            {
                #pragma omp for reduction(+:correct)
                for (int b = 0; b < batch_size; ++b) {
                    int predicted_class = 0;
                    float max_prob = predictions[b * 10];
                    for (int i = 1; i < 10; ++i) {
                        float prob = predictions[b * 10 + i];
                        if (prob > max_prob) {
                            max_prob = prob;
                            predicted_class = i;
                        }
                    }
                    // Find true class
                    int true_class = 0;
                    for (int i = 0; i < 10; ++i) {
                        if (labels[b][i] == 1) {
                            true_class = i;
                            break;
                        }
                    }
                    if (predicted_class == true_class) {
                        correct++;
                    }
                }
            }
            return static_cast<float>(correct) / batch_size;
        }

    public:
        Train(const DataSet& data_set, float learning_rate) 
            : data_set(data_set), 
            conv_layer_1(3, 32, 3, 1, 1, 2, 2, learning_rate), 
            conv_layer_2(32, 64, 3, 1, 1, 2, 2, learning_rate),
            conv_layer_3(64, 128, 3, 1, 1, 2, 2, learning_rate),
            fully_connected_layer(128 * 4 * 4, 10, learning_rate)
            
        {}
        
        void train(int num_epochs, int batch_size, float learning_rate) {
            std::cout << "Starting training with:" << std::endl
                  << "Epochs: " << num_epochs << std::endl
                  << "Batch size: " << batch_size << std::endl
                  << "Learning rate: " << learning_rate << std::endl;
            
            int num_batches = data_set.getNumBatches(batch_size);
            for (int epoch = 0; epoch < num_epochs; ++epoch) {
                float epoch_loss = 0.0f;
                float epoch_accuracy = 0.0f;
                for (int batch = 0; batch < num_batches; ++batch) {
                    // Get batch data
                    const auto& batch_images = data_set.getBatchImages(batch, batch_size);
                    const auto& batch_labels = data_set.getBatchLabels(batch, batch_size);
        
                
                    // Forward pass
                    auto conv_out_1 = conv_layer_1.forward(batch_images, batch_size, 32, 32);
                    auto conv_out_2 = conv_layer_2.forward(conv_out_1, batch_size, 16, 16);
                    auto conv_out_3 = conv_layer_3.forward(conv_out_2, batch_size, 8, 8);
                    auto predictions = fully_connected_layer.forward(conv_out_3, batch_size);
                
                    // Compute metrics
                    float batch_loss = fully_connected_layer.compute_loss(batch_labels);
                    float batch_accuracy = calculate_accuracy(predictions, batch_labels, batch_size);

                    epoch_loss += batch_loss;
                    epoch_accuracy += batch_accuracy;

                    // Backward pass
                    auto fc_grad = fully_connected_layer.backward(batch_labels, batch_size);
                    auto grad_3 = conv_layer_3.backward(fc_grad);
                    auto grad_2 = conv_layer_2.backward(grad_3);
                    auto grad_1 = conv_layer_1.backward(grad_2);
                }
                epoch_loss /= num_batches;
                epoch_accuracy /= num_batches;
                
                std::cout << "\nEpoch " << epoch + 1 << "/" << num_epochs 
                        << "\nAverage Loss: " << epoch_loss
                        << "\nAverage Accuracy: " << (epoch_accuracy * 100.0f) << "%"
                        << std::endl;
            }
        }
};

class PerformanceAnalyzer {
private:
    std::vector<double> times;
    std::vector<int> thread_configs;
    const int num_epochs = 1;
    const int batch_size = 64;
    const float learning_rate = 0.005;
    
public:
    void run_benchmarks(Train& trainer, int min_threads, int max_threads) {
        for(int num_threads = min_threads; num_threads <= max_threads; num_threads *= 2) {
            std::cout << "\nRunning with " << num_threads << " threads...\n";
            omp_set_num_threads(num_threads);
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // Run your training
            trainer.train(num_epochs, batch_size, learning_rate);
            
            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration_cast<std::chrono::milliseconds>
                            (end - start).count();
            
            
            times.push_back(duration);
            thread_configs.push_back(num_threads);
        }
        
        print_analysis();
    }
    
    void print_analysis() {
        std::cout << "\n=== Performance Analysis ===\n";
        double sequential_time = times[0];
        
        for(size_t i = 0; i < times.size(); i++) {
            int threads = thread_configs[i];
            double speedup = sequential_time / times[i];
            double efficiency = speedup / threads;
            
            std::cout << std::fixed << std::setprecision(2)
                     << "Threads: " << threads 
                     << "\nTime: " << times[i] << "ms"
                     << "\nSpeedup: " << speedup << "x"
                     << "\nEfficiency: " << efficiency * 100 << "%\n\n";
        }
    }
};

// Usage in main:
int main() {
    try {
        DataSet data_set("../data");
        data_set.load_data();

        float learning_rate = 0.005;
        Train trainer(data_set, learning_rate);
        PerformanceAnalyzer analyzer;
        
        
        // Test with 1, 2, 4, 8, 10, 12 threads
        analyzer.run_benchmarks(trainer, 8, 16);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

// int main(int argc, char* argv[]) {
//     try {
//         // TIMER START
//         std::cout << "Starting Loading Data..." << std::endl;
//         DataSet data_set("../data");
//         data_set.load_data();
//         std::cout << "Data Loaded Successfully!" << std::endl;

//         int batch_size = 64;
//         if (argc > 1) {
//             batch_size = std::stoi(argv[1]);
//         }

//         float learning_rate = 0.005;
//         int num_epochs = 1;

//         for (int i = 0; i < 8; i++) {
//             omp_set_num_threads(i);
//             auto start_time = std::chrono::high_resolution_clock::now();
//             Train trainer(data_set, learning_rate);
//             trainer.train(num_epochs, batch_size, learning_rate);
//             auto end_time = std::chrono::high_resolution_clock::now();
//             auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
//             std::cout << "Training time: " << duration.count() << " milliseconds" << std::endl;
//             times.push_back(duration.count());
//         }
//         analyze_performance(times);
        

//         // auto start_time = std::chrono::high_resolution_clock::now();
//         // Train trainer(data_set, learning_rate);
//         // trainer.train(num_epochs, batch_size, learning_rate);
//         // // TIMER END
//         // auto end_time = std::chrono::high_resolution_clock::now();
//         // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
//         // std::cout << "Training time: " << duration.count() << " milliseconds" << std::endl;

//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }
//     return 0;
// }