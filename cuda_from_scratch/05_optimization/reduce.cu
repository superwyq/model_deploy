#include <iostream>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "../Timer.hpp"

#define BLOCK_SIZE 512
#define GRID_SIZE 8
/*
 * 规约reduce，指的是将由多个元素组成的向量，经过二元操作符，最终得到一个标量的过程，这个二元操作符可以是加法、乘法、最大值、最小值等
 * 本例实现一个求和规约。
 */

 __global__ void sum_kernel(int *data, int n) {
     int index = blockIdx.x * blockDim.x + threadIdx.x;
     __shared__ int sdata[BLOCK_SIZE]; //共享内存
     int sdata_temp = 0;
     for(int idx = index; idx < n; idx += blockDim.x * gridDim.x) { 
        // 数据量大于线程数，所以在读取这一步的时候就先把数据加起来
        // 例如1024的blocksize，数据量为4096，那就在读取的时候先把i, i+1024, i+2048, i+3072加起来
        // 这样就可以保证数据量等于blocksize，只不过每个shared memory里面的数据加的数量可能不一样
        // 如果是4097个数据，那shared memory[0]就多加了一个数据，但是不影响结果
         sdata_temp += data[idx];
     }
    sdata[threadIdx.x] = sdata_temp;
    __syncthreads(); //等待所有线程都把数据加完

    for(int i = blockDim.x / 2; i >= 1; i /= 2) { 
        //等到都加载完，然后开始规约，每次用剩余数据量一半的线程数，把数据加起来
        int temp = 0;
        if(threadIdx.x < i) { //这里是为了保证不会越界
            // 剩余2i个数据时，将第threadIdx.x个数据和第threadIdx.x + i个数据相加，结果放在第threadIdx.x个数据中
            temp = sdata[threadIdx.x] + sdata[threadIdx.x + i];
            // 这里注意，这种又读又写的操作，最好加一个同步，否则可能会出现数据不一致的情况
        }
        if(threadIdx.x < i) {
            sdata[threadIdx.x] = temp;
        }
    }
    if(blockIdx.x * blockDim.x < n){ //避免出现多余线程的情况
        if(threadIdx.x == 0) {
            atomicAdd(&data[0], sdata[0]); 
            //将每个block的结果加到第一个block的第一个线程上
            // 这里使用了原子操作，因为可能有多个block同时在写入第一个block的数据，所以需要保证数据的一致性
        }
    }
 }

 void sum_cpu(int *data, int n, int *result) {
    for(int i = 0; i < n; i++) {
        *result += data[i];
    }
 }


 int main(){
    const int n = 1 << 18;
    int *data = new int[n];
    int result = 0;
    for(int i = 0; i < n; i++) {
        data[i] = 1;
    }
    Timer timer;
    timer.start();
    for(int i = 0; i < 20; ++i){
        result = 0;
        sum_cpu(data, n, &result);
    }
    timer.end();
    std::cout << "Result:" << result << std::endl;

    timer.start();

    int *d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, data, n*sizeof(int), cudaMemcpyHostToDevice);
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(GRID_SIZE);

    cudaMemcpy(d_data, data, sizeof(int), cudaMemcpyHostToDevice);
    sum_kernel<<<dimGrid, dimBlock>>>(d_data, n);


    cudaMemcpy(data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    timer.end();
    std::cout << "Result:" << data[0] << std::endl;

    timer.start();
    sum_kernel<<<dimGrid, dimBlock>>>(data, n);
    timer.end();
    std::cout << "Result:" << data[0] << std::endl;

    delete[] data;

    return 0;
 }