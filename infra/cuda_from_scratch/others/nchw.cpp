#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

int main(){
    cv::Mat img = cv::imread("/home/wyq/hobby/model_deploy/source/test.jpg", cv::IMREAD_COLOR);
    cv::imshow("img", img);
    if( cv::waitKey(0) == 27 ) return 0;
    int height = img.rows;
    int width = img.cols;
    int channel = img.channels();
    std::cout << "height: " << height << " width: " << width << " channel: " << channel << std::endl;
    int* data = (int*)img.data;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            for(int c = 0; c < channel; c++){
                std::cout << data[i*width*channel + j*channel + c] << " ";
            }
            std::cout << std::endl;
        }
    }
}