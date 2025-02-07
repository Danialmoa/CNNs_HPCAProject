#!/bin/bash
nvcc -o cpp_cuda \
    main.cu \
    dataset.cu \
    conv_block.cu \
    fully_connected.cu \
    adam_optimizer.cu \
    -I./include \
    -arch=sm_75 \
    -std=c++14 \
    -O3
