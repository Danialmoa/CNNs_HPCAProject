cmake_minimum_required(VERSION 3.8)
project(CNNProject CUDA CXX)

# Set CUDA flags and options
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly set CUDA flags for T4
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")

find_package(CUDA REQUIRED)
find_package(OpenMP)

# Add include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/CNNs_HPCAProject/cuda/include
    ${CUDA_INCLUDE_DIRS}
)

# Collect source files
file(GLOB SOURCES 
    "${CMAKE_SOURCE_DIR}/CNNs_HPCAProject/cuda/*.cu"
    "${CMAKE_SOURCE_DIR}/CNNs_HPCAProject/cuda/*.cpp"
)

# Create executable
cuda_add_executable(cnn_train ${SOURCES})

# Link libraries - combine all libraries in a single command
if(OpenMP_CXX_FOUND)
    target_link_libraries(cnn_train 
        PUBLIC 
        ${CUDA_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(cnn_train 
        PUBLIC 
        ${CUDA_LIBRARIES}
    )
endif()

# Set CUDA properties
set_target_properties(cnn_train PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Explicitly set the architecture for each target
set_target_properties(cnn_train PROPERTIES 
    CUDA_ARCHITECTURES "75"
)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)