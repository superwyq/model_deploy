#include <cuda_runtime.h>
#include <iostream>

// Kernel function to perform convolution
//code by copilot
__global__ void convolutionKernel(const float* input, const float* kernel, float* output, int inputSize, int kernelSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < inputSize) {
        float result = 0.0f;
        int halfKernelSize = kernelSize / 2;

        for (int i = -halfKernelSize; i <= halfKernelSize; i++) {
            int inputIndex = tid + i;
            int kernelIndex = halfKernelSize - i;

            if (inputIndex >= 0 && inputIndex < inputSize) {
                result += input[inputIndex] * kernel[kernelIndex];
            }
        }

        output[tid] = result;
    }
}



int main() {
    // Input data
    const int inputSize = 100;
    const int kernelSize = 5;
    float input[inputSize];
    float kernel[kernelSize];
    float output[inputSize];

    // Initialize input and kernel data
    for (int i = 0; i < inputSize; i++) {
        input[i] = i;
    }

    for (int i = 0; i < kernelSize; i++) {
        kernel[i] = 1.0f;
    }

    // Allocate device memory
    float* d_input;
    float* d_kernel;
    float* d_output;
    cudaMalloc((void**)&d_input, inputSize * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernelSize * sizeof(float));
    cudaMalloc((void**)&d_output, inputSize * sizeof(float));

    // Copy input and kernel data to device memory
    cudaMemcpy(d_input, input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (inputSize + blockSize - 1) / blockSize;
    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, inputSize, kernelSize);

    // Copy output data from device to host memory
    cudaMemcpy(output, d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output
    for (int i = 0; i < inputSize; i++) {
        std::cout << output[i] << " ";
        if(i % 10 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}
