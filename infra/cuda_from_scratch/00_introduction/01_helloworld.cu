#include<stdio.h>

__global__ void hello_from_gpu()
{
    printf("hello world from gpu\n");
}

int main(void)
{
    hello_from_gpu<<<4,4>>>();
    cudaDeviceSynchronize();

    return 0;
}
