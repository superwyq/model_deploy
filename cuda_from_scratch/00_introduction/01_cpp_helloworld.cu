#include <iostream>

__global__ void helloWorld() {
    std::cout << "Hello, World!" << std::endl;
}

int main() {
    helloWorld<<<2, 2>>>();
    cudaDeviceSynchronize(); //用于设备函数执行对齐。
    return 0;
}
