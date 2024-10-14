#include <iostream>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <chrono>
#include <cudnn.h>

__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
    int idx = threadIdx.x * blockDim.y + threadIdx.y;
    if (idx >= rows) return;
    float max_value = 0;
    float exp_sum = 0;
    __shared__ float sdata[256];
    for(int i=0; i < cols; i++){
        sdata[i] = input[idx * cols + i];
        max_value = max(max_value, sdata[i]);
    }
    for (int i = 0; i < cols; i++) {
        output[idx * cols + i] = exp(sdata[i]-max_value);
        exp_sum += output[idx * cols + i];
    }
    for (int i = 0; i < cols; i++) {
        output[idx * cols + i] /= exp_sum;
    }

}

void softmax_v1(float* input, float* output, int rows, int cols,int axis=-1) {
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));
    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimGrid(1,1);
    dim3 dimBlock((rows + 255) / 256, 256);
    softmax_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, rows, cols);
    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_input);
    cudaFree(d_output);
}

void softmax_cudnn(float* input, float* output, int rows, int cols) {
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_ACCURATE;
    cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_CHANNEL;
    float alpha = 1.0;
    float beta = 0.0;

    cudnnCreate(&cudnn);
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, rows, cols, 1, 1);

    float *d_input, *d_output;
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));
    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    cudnnSoftmaxForward(cudnn, algo, mode, &alpha, input_desc, d_input, &beta, input_desc, d_output);
    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudnnDestroy(cudnn);
    cudnnDestroyTensorDescriptor(input_desc);
    cudaFree(d_input);
    cudaFree(d_output);
}

bool if_right(float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += output[i * cols + j];
        }
        if (abs(sum - 1) > 1e-6) {
            return false;
        }
    }
    return true;
}

int main() {
    Eigen::MatrixXf input(3, 4);
    input << 1, 2, 3, 4,
             2, 3, 4, 5,
             3, 4, 5, 6;
    Eigen::MatrixXf output(3, 4);
    auto start = std::chrono::high_resolution_clock::now();
    softmax_v1(input.data(), output.data(), 3, 4);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() * 1000 << "ms" << std::endl;
    std::cout << output << std::endl;
    std::cout << if_right(output.data(), 3, 4) << std::endl;

    Eigen::MatrixXf output_cudnn(3, 4);
    start = std::chrono::high_resolution_clock::now();
    softmax_cudnn(input.data(), output_cudnn.data(), 3, 4);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Time: " << diff.count() * 1000 << "ms" << std::endl;
    std::cout << output_cudnn << std::endl;
    std::cout << if_right(output_cudnn.data(), 3, 4) << std::endl;

    return 0;
}