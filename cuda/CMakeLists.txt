cmake_minimum_required(VERSION 3.18)  # Updated minimum version
project(CNNProject CUDA CXX)

# Set CUDA flags and options
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly set CUDA flags for T4
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")

# Modern CUDA finding
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

# Add include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/CNNs_HPCAProject/cuda/include
)

# Collect source files
file(GLOB SOURCES 
    "${CMAKE_SOURCE_DIR}/CNNs_HPCAProject/cuda/*.cu"
    "${CMAKE_SOURCE_DIR}/CNNs_HPCAProject/cuda/*.cpp"
)

# Create executable using modern add_executable
add_executable(cnn_train ${SOURCES})

# Set language for target
set_target_properties(cnn_train PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(cnn_train PRIVATE 
        CUDA::cudart
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(cnn_train PRIVATE 
        CUDA::cudart
    )
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)