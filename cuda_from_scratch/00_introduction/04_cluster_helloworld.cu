#include<stdio.h>

__global__ void __cluster_dims__(2,1,1) hello_from_gpu()
{
    printf("Hello World from GPU!\n");
}

int main()
{
    hello_from_gpu<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}