cmake_minimum_required(VERSION 3.8)
project(SimpleNeuralNetworkGPU CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Add debug flags for CUDA and C++
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")


# Find CUDA
find_package(CUDA REQUIRED)

# Add executable
cuda_add_executable(cpp_cuda
    main.cu
    dataset.cu
    conv_block.cu
    fully_connected.cu
    adam_optimizer.cu
)

# Include directories
target_include_directories(cpp_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set CUDA flags
set_target_properties(cpp_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optional: Set specific GPU architecture
set_property(TARGET cpp_cuda PROPERTY CUDA_ARCHITECTURES 75) 