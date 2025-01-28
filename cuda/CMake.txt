cmake_minimum_required(VERSION 3.8)
project(SimpleNeuralNetworkGPU CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Add executable
cuda_add_executable(cuda
    main.cu
    dataset.cu
    conv_block.cu
    fully_connected.cu
    adam_optimizer.cu
)

# Include directories
target_include_directories(cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set CUDA flags
set_target_properties(cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optional: Set specific GPU architecture
set_property(TARGET cuda PROPERTY CUDA_ARCHITECTURES 75) 