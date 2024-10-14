#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#define OFFSET(channel,element_index,matrix_num)((channel)*(matrix_num)+element_index) //计算矩阵的偏移量

template<typename T>
void cpu_convolution(T* input,T* output,T* kernel,int input_channel,int input_height,int input_width,int kernel_channel,int kernel_height,int kernel_width){
    int output_channel = input_channel;
    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;
    for(int channel = 0;channel < output_channel;channel++){
        for(int row = 0;row < output_height;row++){
            for(int col = 0;col < output_width;col++){
                T sum = 0;
                for(int k_channel = 0;k_channel < kernel_channel;k_channel++){
                    for(int k_row = 0;k_row < kernel_height;k_row++){
                        for(int k_col = 0;k_col < kernel_width;k_col++){
                            sum += input[OFFSET(channel,row+k_row,input_width) + k_col] * kernel[OFFSET(k_channel,k_row,kernel_width) + k_col];
                        }
                    }
                }
                output[OFFSET(channel,row,output_width) + col] = sum;
            }
        }
    }

}

int main(){
    cv::Mat img = cv::imread("/home/wyq/hobby/cuda_from_scratch/test.jpg");
    cv::Mat img_gray;
    cv::cvtColor(img,img_gray,cv::COLOR_BGR2GRAY);
    cv::Mat img_float;
    img_gray.convertTo(img_float,CV_32F);
    cv::Mat kernel = (cv::Mat_<float>(3,3)<<0,-1,0,-1,5,-1,0,-1,0);
    cv::Mat img_conv;
    cpu_convolution((float*)img_float.data,(float*)img_conv.data,(float*)kernel.data,1,img_float.rows,img_float.cols,1,3,3);
    cv::filter2D(img_float,img_conv,-1,kernel);
    cv::imshow("img",img_float);
    cv::imshow("img_conv",img_conv);
    cv::waitKey(0);
    return 0;
}