#include<iostream>
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
    printf("%f \n",C[i]);
}

int main()
{
    int dev = 0;
    cudaSetDevice(dev);

    int N = 5;
    float A[N], B[N], C[N];
    // Initialize input vectors
    for(int i = 0; i < N; ++i)
    {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    // Allocate vectors in device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*sizeof(float));
    cudaMalloc(&d_B, N*sizeof(float));
    cudaMalloc(&d_C, N*sizeof(float));
    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);

    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // Free host memory

    return 0;
}
