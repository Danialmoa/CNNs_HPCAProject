#!/bin/bash
nvcc -o cpp_cuda \
    main.cu \
    dataset.cu \
    conv_block.cu \
    fully_connected.cu \
    adam_optimizer.cu \
    kernels/conv_kernels.cu \
    kernels/activation_kernels.cu \
    kernels/pooling_kernels.cu \
    kernels/batchnorm_kernels.cu \
    -I./include \
    -arch=sm_52 \
    -std=c++14 \
    -O3
