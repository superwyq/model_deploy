#include "cuda_runtime.h"
#include <iostream>

#define SPLIT 4

__global__ void vector_add(float *a, float *b, float *c, int width)
{
    if(threadIdx.x >= width) return;
    for(int i = 0; i < width; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int width = 1 << 15;
    float *a = new float[width];
    float *b = new float[width];
    float *c = new float[width];

    float *c_1 = c;
    float *c_2 = c + width / SPLIT;
    float *c_3 = c + 2 * width / SPLIT;

    float *d_a, *d_b, *d_c, *d_c_1, *d_c_2, *d_c_3;
    cudaMalloc((void **)&d_a, width * sizeof(float));
    cudaMalloc((void **)&d_b, width * sizeof(float));
    cudaMalloc((void **)&d_c, width * sizeof(float));
    cudaMalloc((void **)&d_c_1, width / SPLIT * sizeof(float));
    cudaMalloc((void **)&d_c_2, width / SPLIT * sizeof(float));
    cudaMalloc((void **)&d_c_3, width / SPLIT * sizeof(float));

    for(int i = 0;i<width; ++i){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    cudaMemcpy(d_a, a, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(1, 1);
    dim3 dimBlock(width, 1);

    cudaEvent_t start, stop;
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);
    vector_add<<<dimGrid, dimBlock, 0, stream1>>>(d_a, d_b, d_c_1, width / SPLIT);
    vector_add<<<dimGrid, dimBlock, 0, stream2>>>(d_a + width / SPLIT, d_b + width / SPLIT, d_c_2, width / SPLIT);
    vector_add<<<dimGrid, dimBlock, 0, stream3>>>(d_a + 2 * width / SPLIT, d_b + 2 * width / SPLIT, d_c_3, width / SPLIT);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed time: " << elapsedTime << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);
    vector_add<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed time: " << elapsedTime << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1);
    cudaFree(d_c_2);
    cudaFree(d_c_3);

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;

}