#include<iostream>
#include<cuda_runtime.h>
#include<Eigen/Dense>
#include<chrono>

#define BLOCK_DIM 1024

__global__ void summax_soft(double *input, double *output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x*blockDim.y + threadIdx.y;
    
}