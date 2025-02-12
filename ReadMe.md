# CNN Implementation: CPU vs GPU Performance Comparison

## Overview
This project implements a Convolutional Neural Network (CNN) in both CPU (C++) and GPU (CUDA) versions to compare their performance. Developed as part of the High Performance Computing course at the University of Siena.

## Dataset
The project uses the CIFAR-10 dataset
## Requirements

- C++ compiler with C++11 or later support
- OpenMP 
- CUDA Toolkit 

## Project Structure
cpu/ # CPU implementation
cuda/ # GPU implementation
results/ # Performance results and visualizations
data/ # Dataset directory

## Usage

### CPU Implementation
g++-14 -std=c++14 -fopenmp -o cpucode  cpuCodeOpenMP.cpp
./cpucode

### GPU Implementation
./compile.sh 
./cpp_cuda

## Author
Danial Moafi
