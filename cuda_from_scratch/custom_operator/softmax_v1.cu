#include <iostream>
#include <cuda_runtime.h>
#include <eigen3/Eigen/Dense>
#include <chrono>

__global__ void softmax_v1(double *input, double *output, int rows, int cols){
    int idx = blockIdx.x * blockDim.x + threadIdx.x * blockDim.y + threadIdx.y;
    if (idx > rows){
        return;
    }
    double max_value = -INFINITY;
    double exp_sum = 0;
    for(int i=0; i < cols; ++i){
        double temp_max = input[idx * cols + i];
        max_value = max(max_value, input[idx * cols + i]);
        exp_sum = exp_sum * (exp(temp_max - max_value)) + exp(input[idx * cols + i] - max_value);
    }
    for(int i=0; i < cols; ++i){
        output[idx * cols + i] = exp(input[idx * cols + i] - max_value) / exp_sum;
    }
}

template <int rows, int cols>
Eigen::Matrix<double, rows, cols> softmax(Eigen::Matrix<double, rows, cols> input){
    Eigen::Matrix<double, rows, cols> output;

    double *d_input,*d_output;
    cudaMalloc((void **)&d_input, rows * cols * sizeof(double));
    cudaMalloc((void **)&d_output, rows * cols * sizeof(double));
    cudaMemcpy(d_input, input.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(1,1);
    dim3 dimBlock((rows + 255) / 256, 256);
    softmax_v1<<<dimGrid, dimBlock>>>(d_input, d_output, rows, cols);
    cudaMemcpy(output.data(), d_output, rows* cols * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_output);
    return output;
}

int main(int argc, char* argv[]){
    const int rows = 16;
    const int cols = 1024;
    Eigen::Matrix<double, rows, cols> input = Eigen::Matrix<double, rows, cols>::Random();
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; ++i){
        Eigen::Matrix<double, rows, cols> output = softmax<rows,cols>(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end - start;
    double elapsed_time_ms = elapsed_time.count();
    std::cout << "Elapsed time: " << elapsed_time_ms << " ms" << std::endl;
}