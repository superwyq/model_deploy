#include <iostream>

__global__ void helloWorld() {
    std::cout << "Hello, World!" << std::endl;
}

int main() {
    helloWorld<<<2, 2>>>();
    cudaDeviceSynchronize();
    return 0;
}
