#include <iostream>

#define SPLIT 4

__global__ void vector_add(float *a, float *b, float *c, int width){
    int index = (blockIdx.x * gridDim.y + blockIdx.y)*blockDim.x * blockDim.y + threadIdx.x * blockDim.y + threadIdx.y;
    // 计算线程的全局索引，第blockIdx.x行第blockIdx.y列的block中的第threadIdx.x行第threadIdx.y列的线程
    if (index >= width) return;
    c[index] = a[index] + b[index];
    
}

void single_stream(float *a, float *b, float *c, int width){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0); //0是指默认流
    cudaEventSynchronize(start);

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, width * sizeof(float));
    cudaMalloc((void **)&d_b, width * sizeof(float));
    cudaMalloc((void **)&d_c, width * sizeof(float));

    cudaMemcpy(d_a, a, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(1, width >> 10);
    dim3 dimBlock(256, 4);
    vector_add<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);
    cudaMemcpy(c, d_c, width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Single stream time: " << time << "ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void multi_stream(float *a, float *b, float *c, int width){
    cudaStream_t stream1, stream2, stream3,stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);

    float *d_a, *d_b, *d_c_1, *d_c_2, *d_c_3, *d_c_4;

    cudaMalloc((void **)&d_a, width * sizeof(float));
    cudaMalloc((void **)&d_b, width * sizeof(float));
    cudaMalloc((void **)&d_c_1, width / SPLIT * sizeof(float));
    cudaMalloc((void **)&d_c_2, width / SPLIT * sizeof(float));
    cudaMalloc((void **)&d_c_3, width / SPLIT * sizeof(float));
    cudaMalloc((void **)&d_c_4, width / SPLIT * sizeof(float));

    dim3 dimGrid(1, 1);
    dim3 dimBlock(width / SPLIT, 1);
    
    cudaMemcpyAsync(d_a, a, width / SPLIT * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, b, width / SPLIT * sizeof(float), cudaMemcpyHostToDevice, stream1);
    vector_add<<<dimGrid, dimBlock, 0, stream1>>>(d_a, d_b, d_c_1, width / SPLIT);
    cudaMemcpyAsync(c, d_c_1, width / SPLIT * sizeof(float), cudaMemcpyDeviceToHost, stream1);

    cudaMemcpyAsync(d_a + width / SPLIT, a + width / SPLIT, width / SPLIT * sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_b + width / SPLIT, b + width / SPLIT, width / SPLIT * sizeof(float), cudaMemcpyHostToDevice, stream2);
    vector_add<<<dimGrid, dimBlock, 0, stream2>>>(d_a + width / SPLIT, d_b + width / SPLIT, d_c_2, width / SPLIT);
    cudaMemcpyAsync(c + width / SPLIT, d_c_2, width / SPLIT * sizeof(float), cudaMemcpyDeviceToHost, stream2);

    cudaMemcpyAsync(d_a + 2 * width / SPLIT, a + 2 * width / SPLIT, width / SPLIT * sizeof(float), cudaMemcpyHostToDevice, stream3);
    cudaMemcpyAsync(d_b + 2 * width / SPLIT, b + 2 * width / SPLIT, width / SPLIT * sizeof(float), cudaMemcpyHostToDevice, stream3);
    vector_add<<<dimGrid, dimBlock, 0, stream3>>>(d_a + 2 * width / SPLIT, d_b + 2 * width / SPLIT, d_c_3, width / SPLIT);
    cudaMemcpyAsync(c + 2 * width / SPLIT, d_c_3, width / SPLIT * sizeof(float), cudaMemcpyDeviceToHost, stream3);

    cudaMemcpyAsync(d_a + 3 * width / SPLIT, a + 3 * width / SPLIT, width / SPLIT * sizeof(float), cudaMemcpyHostToDevice, stream4);
    cudaMemcpyAsync(d_b + 3 * width / SPLIT, b + 3 * width / SPLIT, width / SPLIT * sizeof(float), cudaMemcpyHostToDevice, stream4);
    vector_add<<<dimGrid, dimBlock, 0, stream4>>>(d_a + 3 * width / SPLIT, d_b + 3 * width / SPLIT, d_c_4, width / SPLIT);
    cudaMemcpyAsync(c + 3 * width / SPLIT, d_c_4, width / SPLIT * sizeof(float), cudaMemcpyDeviceToHost, stream4);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    cudaStreamSynchronize(stream4);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Multi stream time: " << time << "ms" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1);
    cudaFree(d_c_2);
    cudaFree(d_c_3);
    cudaFree(d_c_4);
}

bool if_right(float *c, int width){
    for(int i = 0; i < width; i++){
        if(c[i] != 3.0f){
            std::cout << "Error at: " << i << std::endl;
            return false;
        }
    }
    return true;
}

int main(){
    int width = 1 << 12;
    float *a = new float[width];
    float *b = new float[width];
    float *c = new float[width];

    for(int i = 0; i < width; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    single_stream(a, b, c, width);
    if_right(c, width);
    multi_stream(a, b, c, width);
    if_right(c, width);

    return 0;
}