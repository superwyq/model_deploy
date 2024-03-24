#include <iostream>
#include <chrono>
// Kernel function for matrix multiplication on the GPU
__global__ void matrixMulGPU(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrixMulCPU(int* A, int* B, int* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


int main() {
    int N = 1000; // Size of the square matrices
    int matrixSize = N * N * sizeof(int);

    // Allocate memory for matrices on the host (CPU)
    int* h_A = (int*)malloc(matrixSize);
    int* h_B = (int*)malloc(matrixSize);
    int* h_C = (int*)malloc(matrixSize);

    // Initialize matrices with some values
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (i + 1) % 10;
        h_B[i] = (i + 1) % 10;
    }

    // Initialize matrices with input values
    // std::cout << "Enter values for matrix A:" << std::endl;
    // for (int i = 0; i < N * N; i++) {
    //     std::cin >> h_A[i];
    // }

    // std::cout << "Enter values for matrix B:" << std::endl;
    // for (int i = 0; i < N * N; i++) {
    //     std::cin >> h_B[i];
    // }

    // Allocate memory for matrices on the device (GPU)
    int* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(2, 2);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    auto start = std::chrono::high_resolution_clock::now();
    // Launch kernel function for matrix multiplication on the GPU
    matrixMulGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "GPU Time taken: " << duration.count() << " ms" << std::endl;


    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_cpu = end_cpu - start_cpu;
    std::cout << "CPU Time taken: " << duration_cpu.count() << " ms" << std::endl;

    // Free memory on the device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free memory on the host
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
