#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

void __global__ add(const double *x, const double *y, double *z, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        z[i] = x[i] + y[i];
    }
}

int main(){

    const int N = 100000000;
    const int M = sizeof(double) * N;
    std::cout << "M: " << M << std::endl;
    double *h_x = (double *)malloc(M);
    double *h_y = (double *)malloc(M);
    double *h_z = (double *)malloc(M);

    double a = 1.1;
    double b = 2.2;
    for(int i = 0; i < N; i++){
        h_x[i] = a;
        h_y[i] = b;
    }

    double *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);
    
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;

    cudaEvent_t start, stop; //定义事件
    float elapsedTime[10]; //时间差
    for(int i = 0; i < 10; i++){
        cudaEventCreate(&start); //创建事件
        cudaEventCreate(&stop);
        cudaEventRecord(start); //记录一个开始事件
        cudaEventSynchronize(start); //同步事件

        add<<<grid_size, block_size>>>(d_x, d_y, d_z, N); //要计时的代码
        cudaDeviceSynchronize();

        cudaEventRecord(stop); //记录一个结束事件
        cudaEventSynchronize(stop); //同步事件

        cudaEventElapsedTime(&elapsedTime[i], start, stop); //计算时间差
        std::cout << "Elapsed time: " << elapsedTime[i] << " ms" << std::endl;
        cudaEventDestroy(start); //销毁事件
        cudaEventDestroy(stop);
    }
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    float average = 0;
    for(int i =0; i< 10; i++){
        average += elapsedTime[i];
    }
    average /= 10.0;
    std::cout << "Average time: " << average << " ms" << std::endl;
    return 0;
}