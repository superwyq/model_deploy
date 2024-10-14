#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>

__global__ void add(float* a, float* b, float* c, int* shape){
    int dx = threadIdx.x;
    int dy = threadIdx.y;
    int width = shape[0];
    int height = shape[1];
    int channel = shape[2];
    int index = dx + dy * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    if (dx < width && dy < height){
        for (int i = 0; i < channel; i++){
            c[index] = a[index] + b[index];
        }
    }
}

void add_cpu(float* a, float* b,float* c,int* shape){
    int width = shape[0];
    int height = shape[1];
    int channel = shape[2];
    for (int i = 0; i < width; i++){
        for (int j = 0; j < height; j++){
            for (int k = 0; k < channel; k++){
                c[i * height * channel + j * channel + k] = a[i * height * channel + j * channel + k] + b[i * height * channel + j * channel + k];
            }
        }
    }
}

__global__ add_gpu_v2(float* a, float* b, float* c, int* shape){
    int dx = threadIdx.x;
    int dy = threadIdx.y;
    int width = shape[0];
    int height = shape[1];
    int channel = shape[2];
    int index = dx + dy * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    __shared__ float a_shared[10][100];
    __shared__ float b_shared[10][100];
    if (dx < width && dy < height){
        a_shared[dx][dy] = a[index];
        b_shared[dx][dy] = b[index];
        __syncthreads();
        for (int i = 0; i < channel; i++){
            c[index] = a_shared[dx][dy] + b_shared[dx][dy];
        }
    }
}

int main(){
    float* a, *b, *c;
    int* shape;
    cudaMallocManaged(&a, 1000 * sizeof(float));
    cudaMallocManaged(&b, 1000 * sizeof(float));
    cudaMallocManaged(&c, 1000 * sizeof(float));
    cudaMallocManaged(&shape, 3 * sizeof(int));
    shape[0] = 10;
    shape[1] = 100;
    shape[2] = 1;
    for (int i = 0; i < 1000; i++){
        a[i] = 1.0;
        b[i] = 2.0;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    add<<<1, dim3(10, 100)>>>(a, b, c, shape);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "cost Time: " << time << "ms" << std::endl;
    for (int i = 0; i < 100; i++){
        std::cout << c[i] << " ";
        if (i % 10 == 9){
            std::cout << std::endl;
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto start_cpu = std::chrono::steady_clock::now();
    add_cpu(a, b, c, shape);
    auto end_cpu = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_cpu - start_cpu;
    std::cout << "CPU cost Time: " << elapsed_seconds.count() * 1000 << "ms" << std::endl;
    for (int i = 0; i < 100; i++){
        std::cout << c[i] << " ";
        if (i % 10 == 9){
            std::cout << std::endl;
        }
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(shape);
    return 0;
}