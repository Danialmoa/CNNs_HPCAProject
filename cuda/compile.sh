#!/bin/bash
nvcc -o cpp_cuda \
    main.cu \
    dataset.cu \
    conv_block.cu \
    fully_connected.cu \
    adam_optimizer.cu \
    -I./include \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lcudnn \
    -arch=sm_52 \
    -std=c++14 \
    -O3
