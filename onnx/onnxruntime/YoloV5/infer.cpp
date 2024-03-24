#include<iostream>
#include<onnxruntime/onnxruntime_cxx_api.h>
#include<opencv2/opencv.hpp>
#include<array>

struct YoloV5
{
    YoloV5() {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
                                                        input_shape_.data(), input_shape_.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
                                                        output_shape_.data(), output_shape_.size());
    }

    std::ptrdiff_t Run() {
        const char* input_names[] = {"Input"};
        const char* output_names[] = {"Output"};

        Ort::RunOptions run_options;
        
        session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
        for(auto i : results_){
            std::cout << i << " ";
        }
        result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
        return result_;
    }

    static constexpr const int width_ = 640;
    static constexpr const int height_ = 640;
    static constexpr const int num_classes_ = 80;
    
    std::array<float, width_ * height_> input_image_;
    std::array<float, 85> results_;
    int64_t result_{0};

    private:
        Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "test"};
        Ort::Session session_{env_, "/home/wyq/hobby/model_deploy/onnx/onnxruntime/YoloV5/yolov5s.onnx",Ort::SessionOptions{nullptr}};
        

        Ort::Value input_tensor_{nullptr};
        std::array<int64_t,4> input_shape_{1, 3, width_, height_};
        
        Ort::Value output_tensor_{nullptr};
        std::array<int64_t,2> output_shape_{1, 10};
};


int main() {
    cv::Mat input_image = cv::imread("test.jpg");
    cv::Mat resize_image;
    const int model_width = 640;
    const int model_height = 640;
    const float ratio = std::min(model_width / (input_image.cols * 1.0f),
                                model_height / (input_image.rows * 1.0f));
    // 等比例缩放
    const int border_width = input_image.cols * ratio;
    const int border_height = input_image.rows * ratio;
    // 计算偏移值
    const int x_offset = (model_width - border_width) / 2;
    const int y_offset = (model_height - border_height) / 2;
    cv::resize(input_image, resize_image, cv::Size(border_width, border_height));
    cv::copyMakeBorder(resize_image, resize_image, y_offset, y_offset, x_offset,
                        x_offset, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    // 转换为RGB格式
    cv::cvtColor(resize_image, resize_image, cv::COLOR_BGR2RGB);
    auto input_blob = new float[model_height * model_width * 3];
    const int channels = resize_image.channels();
    const int width = resize_image.cols;
    const int height = resize_image.rows;
    for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
        input_blob[c * width * height + h * width + w] =
            resize_image.at<cv::Vec3b>(h, w)[c] / 255.0f;
        }
    }
    }
    YoloV5 yolo;
    cv::Mat img = cv::imread("/home/wyq/hobby/model_deploy/test.jpg");
    cv::resize(img, img, cv::Size(yolo.width_, yolo.height_));

    yolo.Run();
    return 0;
}