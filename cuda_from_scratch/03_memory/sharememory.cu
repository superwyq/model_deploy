#include <cuda_runtime.h>

#include <iostream>

#define TILE_WIDTH 16

__global__ void matrixMul(int *a, int *b, int *c, int width) {
  __shared__ int subTileA[TILE_WIDTH][TILE_WIDTH]; //初始化shared memory中的变量
  __shared__ int subTileB[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  int sum = 0;
  for (int t = 0; t < width / TILE_WIDTH; ++t) {
    subTileA[threadIdx.y][threadIdx.x] =
        a[row * width + t * TILE_WIDTH + threadIdx.x];
    subTileB[threadIdx.y][threadIdx.x] =
        b[(t * TILE_WIDTH + threadIdx.y) * width + col];
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      sum += subTileA[threadIdx.y][k] * subTileB[k][threadIdx.x];
    }
    __syncthreads();
  }

  c[row * width + col] = sum;
}

int main() {
  const int width = 1024;
  const int size = width * width * sizeof(int);

  int *h_a, *h_b, *h_c;
  h_a = new int[width * width];
  h_b = new int[width * width];
  h_c = new int[width * width];

  // 初始化 h_a 和 h_b

  int *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  dim3 dimGrid(width / TILE_WIDTH, width / TILE_WIDTH);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);

  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  // 处理结果

  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
