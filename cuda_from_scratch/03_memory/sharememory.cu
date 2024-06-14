#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_WIDTH 16

__global__ void matrixMul(int *a, int *b, int *c, int width) {
  __shared__ int subTileA[BLOCK_WIDTH][BLOCK_WIDTH]; //share memory是可以声明在核函数内的
  __shared__ int subTileB[BLOCK_WIDTH][BLOCK_WIDTH];

  int row = blockIdx.y * BLOCK_WIDTH + threadIdx.y;
  int col = blockIdx.x * BLOCK_WIDTH + threadIdx.x;

  int sum = 0;
  
  for (int t = 0; t < width / BLOCK_WIDTH; ++t) { //初始化
    subTileA[threadIdx.y][threadIdx.x] =
        a[row * width + t * BLOCK_WIDTH + threadIdx.x];
    subTileB[threadIdx.y][threadIdx.x] =
        b[(t * BLOCK_WIDTH + threadIdx.y) * width + col];
    __syncthreads();

    for (int k = 0; k < BLOCK_WIDTH; ++k) {
      sum += subTileA[threadIdx.y][k] * subTileB[k][threadIdx.x];
    }
    __syncthreads();
  }

  c[row * width + col] = sum;
}

__global__ void matrixMul_worse(int *a, int *b, int *c, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int sum = 0;
  for (int i = 0; i < width; ++i) {
    sum += a[row * width + i] * b[i * width + col];
  }

  c[row * width + col] = sum;
}

int main() {
    const int width = 128;
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
    cudaEventRecord(start); //记录一个开始事件
    cudaEventSynchronize(start); //同步事件
    matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);
    cudaDeviceSynchronize();

    cudaEventRecord(stop); //记录一个结束事件
    cudaEventSynchronize(stop); //同步事件

    cudaEventElapsedTime(&runtime[0], start, stop); //计算时间差
    std::cout << "Elapsed time 1: " << runtime[0] << " ms" << std::endl;
    cudaEventDestroy(start); //销毁事件
    cudaEventDestroy(stop);

    cudaEventCreate(&start); //创建事件
    cudaEventCreate(&stop);
    cudaEventRecord(start); //记录一个开始事件
    cudaEventSynchronize(start); //同步事件
    matrixMul_worse<<<dimGrid, dimBlock>>>(d_a, d_b, d_c2, width);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); //记录一个结束事件
    cudaEventSynchronize(stop); //同步事件
    cudaEventElapsedTime(&runtime[1], start, stop); //计算时间差
    std::cout << "Elapsed time 2: " << runtime[1] << " ms" << std::endl;

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c2, d_c2, size, cudaMemcpyDeviceToHost);

    for (int i = 1; i < width; ++i) {
      std::cout << h_c[i*width]<< " ";
      if (i % 16 == 0) {
        std::cout << std::endl;
      }
    }

    for (int i = 0; i < width; ++i) {
      if (h_c[i] != h_c2[i]) {
        std::cout << "Error" << std::endl;
        break;
      }
    }

    // 处理结果

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
