#include<stdio.h>
#include<cuda_runtime.h>


__global__ void build_in_variables(void)
{
    // build-in variables
    // blockDim：等同于threadsPerBlock
    // gridDim：等同于numBlocks
    // blockIdx:一个block在grid中的id
    // threadIdx:一个thread在block中的id

    const int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    const int threadId = threadIdx.x + blockDim.x * threadIdx.y;

    printf("blockIdx=(%d,%d) \n",blockIdx.x,blockIdx.y);
    printf("threadIdx=(%d,%d) \n",threadIdx.x,threadIdx.y);
    printf("blockid=:%d,threadId=%d \n",blockId,threadId);

}

int main(void)
{
    printf("*****device message*******\n");

    int dev=0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Using Device %d:%s\n",dev,deviceProp.name);
    printf("Device %d has compute capability %d.%d.\n",dev,deviceProp.major,deviceProp.minor);
    printf("Device %d has %d multi-processors.\n",dev,deviceProp.multiProcessorCount);
    printf("Device %d has %zu byte total global memory.\n",dev,deviceProp.totalGlobalMem);
    printf("Device %d has %zu byte total constant memory.\n",dev,deviceProp.totalConstMem);
    printf("Device %d has %zu byte shared memory per block.\n",dev,deviceProp.sharedMemPerBlock);
    printf("Device %d has %d total registers per block.\n",dev,deviceProp.regsPerBlock);
    printf("Device %d has %d max threads per block.\n",dev,deviceProp.maxThreadsPerBlock);
    printf("Device %d has %d max threads dimensions.\n",dev,deviceProp.maxThreadsDim[0]);
    printf("Device %d has %u max grid size.\n",dev,deviceProp.maxGridSize[0]);
    printf("Device %d has %d warp size.\n",dev,deviceProp.warpSize);
    printf("Device %d has %d clock rate.\n",dev,deviceProp.clockRate);
    printf("Device %d has %d max threads per multi-processor.\n",dev,deviceProp.maxThreadsPerMultiProcessor);
    

    dim3 numBlocks(2,2);
    dim3 threadsPerBlock(2,2);
    build_in_variables<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceReset();
    return 0;
}