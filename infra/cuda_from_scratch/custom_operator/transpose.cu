#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;

__global__ void transpose_kernel(unsigned char *src, unsigned char *dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int src_idx = y * width + x;
        int dst_idx = x * height + y;
        dst[dst_idx] = src[src_idx];
    }
}

void transpose_image(unsigned char *h_src, unsigned char *h_dst, int width, int height) {
    // 计算线程块大小和网格大小
    dim3 block(16, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 分配 GPU 内存
    unsigned char *d_src, *d_dst;
    cudaMalloc(&d_src, width * height * sizeof(unsigned char));
    cudaMalloc(&d_dst, width * height * sizeof(unsigned char));

    // 从 CPU 复制到 GPU
    cudaMemcpy(d_src, h_src, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 调用核函数
    transpose_kernel<<<grid, block>>>(d_src, d_dst, width, height);

    // 从 GPU 复制回 CPU
    cudaMemcpy(h_dst, d_dst, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_src);
    cudaFree(d_dst);
}

int main()
{
    cv::Mat src = cv::imread("/home/wyq/hobby/model_deploy/source/dog.jpeg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Can't load image" << std::endl;
        return -1;
    }
    transpose_image(src.data, src.data, src.cols, src.rows);
    cv::imshow("src", src);
    cv::waitKey();
    return 0;
}