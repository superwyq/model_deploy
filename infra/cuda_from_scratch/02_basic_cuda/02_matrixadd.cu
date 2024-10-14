#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "error_check.cuh"
#include <iostream>
#include <stdio.h>

void __global__ add(const double *x, const double *y, double *z, int N);
//global修饰，host调用，device执行,不能有返回值
double __device__ __host__ add_device(const double x, const double y) { return x + y;  }
// device和host可以同时修饰一个函数，使得这个函数既是一个普通C++函数，又是一个设备函数
// device修饰，设备函数，device调用，device执行，可以有返回值

int main(void) {
    const int N = 1 << 10;
    const int M = sizeof(double) * N;
    double *h_x = (double *)malloc(M);
    double *h_y = (double *)malloc(M);
    double *h_z = (double *)malloc(M);

    const double a = 1.23;
    const double b = 4.56;

    for (int i = 0; i < N; i++) {
        h_x[i] = a;
        h_y[i] = b;
    }

    double *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, M);
    //在调用第一个API时，设备会隐式初始化
    cudaMalloc((void **)&d_y, M);
    //双重指针，因为cudamalloc是分配内存并将地址指针保存在d_y指向的地方中，而不是直接将d_y指向地址
    cudaMalloc((void **)&d_z, M);
    //这么做的原因是因为cuda运行时函数的返回值是用来返回错误代号的，成功则返回cudaSuccess
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    //内存复制，第一个是目标地址，第二个是源地址，第三个是字节大小，第四个是方向
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);
    
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size; //确保不会因为四舍五入导致的grid_size不够大
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    CHECK(cudaGetLastError());
    //对于核函数，因为不返回任何值，所以需要调用cudaGetLastError函数来收集错误信息然后传递给CHECK
    CHECK(cudaDeviceSynchronize());
    // 设备函数同步，类似于join

    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));

    for (int i = 1; i < N+1; i++) {
        std::cout << h_z[i-1] << "\t";
        if (i % 4 == 0) {
            std::cout << "\n";
        }
    }
    std::cout << std::endl;
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x); //谁分配谁释放，cuda也一样
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}

void __global__ add(const double *x, const double *y, double *z, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // cuda内置变量，blockIdx表示当下的blockid，blockDim表示每个block的维度，threadIdx表示当下thread在block中的threadid
    if (i > N) return;
    //核函数不能返回值，但是使用return来结束还是可以的
    z[i] = add_device(x[i], y[i]);
}