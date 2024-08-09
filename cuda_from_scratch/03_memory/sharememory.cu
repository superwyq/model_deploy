#include <cuda_runtime.h>
#include <iostream>
#include "nvToolsExt.h"

#define BLOCK_WIDTH 64

__global__ void matrixMul(int *a, int *b, int *c, int width) {
  __shared__ int subTileA[BLOCK_WIDTH][BLOCK_WIDTH]; //share memory是可以声明在核函数内的
  __shared__ int subTileB[BLOCK_WIDTH][BLOCK_WIDTH];

  int row = blockIdx.y * BLOCK_WIDTH + threadIdx.y;
  int col = blockIdx.x * BLOCK_WIDTH + threadIdx.x;

  if(row >= width || col >= width) return;

  for (int i = 0; i < width / BLOCK_WIDTH; ++i) {
    subTileA[threadIdx.y][threadIdx.x] = a[row * width + i * BLOCK_WIDTH + threadIdx.x];
    subTileB[threadIdx.y][threadIdx.x] = b[(i * BLOCK_WIDTH + threadIdx.y) * width + col];
    __syncthreads();

    int sum = 0;
    for (int j = 0; j < BLOCK_WIDTH; ++j) {
      sum += subTileA[threadIdx.y][j] * subTileB[j][threadIdx.x];
    }
    __syncthreads();
    c[row * width + col] += sum;
  }
}

__global__ void matrixMul_worse(int *a, int *b, int *c, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;


  if(row >= width || col >= width) return;
  int sum = 0;
  for (int i = 0; i < width; ++i) {
    sum += a[row * width + i] * b[i * width + col];
  }

  c[row * width + col] = sum;
}

int main() {
    const int width = 1 << 10;
    const int size = width * width * sizeof(int);

    int *h_a, *h_b, *h_c, *h_c2;
    h_a = new int[width * width];
    h_b = new int[width * width];
    h_c = new int[width * width];
    h_c2 = new int[width * width];

    // 初始化 h_a 和 h_b
    for (int i = 0; i < width; ++i) {
        h_a[i*width] = 2;
        h_b[i*width] = 2;
    }

    int *d_a, *d_b, *d_c, *d_c2;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    cudaMalloc((void **)&d_c2, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(width / BLOCK_WIDTH, width / BLOCK_WIDTH);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

    float runtime[2];
    cudaEvent_t start, stop; //定义事件
    cudaEventCreate(&start); //创建事件
    cudaEventCreate(&stop);
    cudaEventRecord(start,0); //记录一个开始事件
    cudaEventSynchronize(start); //同步事件

    matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);
    cudaStreamSynchronize(0);
    cudaEventRecord(stop,0); //记录一个结束事件
    cudaEventSynchronize(stop); //同步事件

    cudaEventElapsedTime(&runtime[0], start, stop); //计算时间差
    std::cout << "Elapsed time 1: \t" << runtime[0] << " ms" << std::endl;

    cudaEvent_t start2, stop2;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEventCreate(&start2); //创建事件
    cudaEventCreate(&stop2);
    cudaEventRecord(start2,stream); //记录一个开始事件
    cudaEventSynchronize(start2); //同步事件

    matrixMul_worse<<<dimGrid, dimBlock, 0, stream>>>(d_a, d_b, d_c2, width);
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop2,stream); //记录一个结束事件
    cudaEventSynchronize(stop2); //同步事件
    cudaEventElapsedTime(&runtime[1], start2, stop2); //计算时间差
    std::cout << "worse Elapsed time 2: \t" << runtime[1] << " ms" << std::endl;


    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c2, d_c2, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < width; ++i) {
      if (h_c[i] != h_c2[i]) {
        std::cout << "Error" << std::endl;
        std::cout << h_c[i] << " " << h_c2[i] << std::endl;
        break;
      }
      std::cout << h_c[i] << " " << h_c2[i];
    }

    // 处理结果

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_c2);

    return 0;
}
