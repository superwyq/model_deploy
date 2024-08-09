#include <iostream>
#include <chrono>
#include <eigen3/Eigen/Dense>

template<int row, int col>
Eigen::Matrix<double,row,col> softmax(Eigen::Matrix<double,row,col> input){
    Eigen::Matrix<double, row, 1> max_value, sum_exp_value;
    Eigen::Matrix<double, row, col> output;
    for(int i = 0; i < input.rows(); ++i){
        double temp_max = -INFINITY, temp_sum = 0;
        for(int j = 0; j < input.cols(); ++j){
            double temp = temp_max;
            temp_max = std::max(temp_max, input(i,j));
            temp_sum = temp_sum*exp(temp - temp_max) + exp(input(i,j) - temp_max);
        }
        max_value(i,0) = temp_max;
        sum_exp_value(i,0) = temp_sum;
    }
    for(int i = 0; i < input.rows(); ++i){
        for(int j = 0; j < input.cols(); ++j){
            output(i,j) = exp(input(i,j) - max_value(i,0))/sum_exp_value(i,0);
        }
    }
    return output;
}

int main(){
    // 不开启优化的情况下，16*1024的矩阵运算耗时约为2.954ms
    // 开启O3优化的情况下，16*1024的矩阵运算耗时约为0.147ms
    Eigen::Matrix<double, 16, 1024> input = Eigen::Matrix<double, 16, 1024>::Random();
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1000; ++i){
        Eigen::Matrix<double, 16, 1024> output = softmax(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Per Elapsed time: " << elapsed.count() << " ms\n";
    return 0;
}