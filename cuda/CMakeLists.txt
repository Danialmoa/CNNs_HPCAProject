cmake_minimum_required(VERSION 3.8)
project(SimpleNeuralNetworkGPU CUDA CXX)

# Force CUDA compiler to use older toolchain
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "11.0")
set(CMAKE_CUDA_ARCHITECTURES 70)  # Using a more compatible architecture
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -arch=sm_70")

find_package(CUDA REQUIRED)

# Add executable
add_executable(cpp_cuda
    main.cu
    dataset.cu
    conv_block.cu
    fully_connected.cu
    adam_optimizer.cu
)

# Set properties
set_target_properties(cpp_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(cpp_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)