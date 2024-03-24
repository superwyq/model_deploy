#include<iostream>
#include<onnxruntime/onnxruntime_cxx_api.h>
#include<array>
#include<cmath>
#include<algorithm>
#include<opencv2/opencv.hpp>

#pragma comment(lib, "onnxruntime.lib")

template<typename T>
static void softmax(T& input) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    float sum = 0.0f;
    for(size_t i=0; i!=input.size(); ++i) {
        y[i] = std::exp(input[i] - rowmax);
        sum += y[i];
    }
    for(size_t i=0; i!=input.size(); ++i) {
        input[i] = y[i] / sum;
    }
}

struct MINIST 
{
    MINIST() {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
                                                        input_shape_.data(), input_shape_.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
                                                        output_shape_.data(), output_shape_.size());
    }

    std::ptrdiff_t Run() {
        const char* input_names[] = {"Input3"};
        const char* output_names[] = {"Plus214_Output_0"};

        Ort::RunOptions run_options;
        
        session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
        softmax(results_);
        for(auto i : results_){
            std::cout << i << " ";
        }
        result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
        return result_;
    }

    static constexpr const int width_ = 28;
    static constexpr const int height_ = 28;
    
    std::array<float, width_ * height_> input_image_;
    std::array<float, 10> results_;
    int64_t result_{0};

    private:
        Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "test"};
        Ort::Session session_{env_, "/home/wyq/hobby/model_deploy/onnx/export_onnx/MINIST/mnist.onnx",Ort::SessionOptions{nullptr}};
        
        Ort::Value input_tensor_{nullptr};
        std::array<int64_t,4> input_shape_{1, 1, width_, height_};
        
        Ort::Value output_tensor_{nullptr};
        std::array<int64_t,2> output_shape_{1, 10};
};

const constexpr int drawing_area_inset_{4};
const constexpr int drawing_area_scale_{4};
const constexpr int drawing_area_width_{MINIST::width_ * drawing_area_scale_};
const constexpr int drawing_area_height_{MINIST::height_ * drawing_area_scale_};

std::unique_ptr<MINIST> minist_;

bool draw_{false};
bool clear_{false};
void draw_callback(int event, int x, int y, int flags, void* userdata) {
    if(event == cv::EVENT_LBUTTONDOWN) {
        draw_ = true;
    } else if(event == cv::EVENT_LBUTTONUP) {
        draw_ = false;
    }else if(event == cv::EVENT_RBUTTONDOWN) {
        clear_ = true;
    } else if(event == cv::EVENT_RBUTTONUP) {
        clear_ = false;
    }
    if(draw_) {
        cv::circle(*static_cast<cv::Mat*>(userdata), cv::Point(x, y), 2, cv::Scalar(255), -1);

    }
    if(clear_) {
        cv::Mat image(drawing_area_height_*2, drawing_area_width_*2, CV_8UC1, cv::Scalar(0));
        *static_cast<cv::Mat*>(userdata) = image;
    }
}
int main(){
    minist_ = std::make_unique<MINIST>();
    cv::Mat image(drawing_area_height_*2, drawing_area_width_*2, CV_8UC1, cv::Scalar(0));
    cv::namedWindow("MINIST", cv::WINDOW_GUI_EXPANDED);
    cv::setMouseCallback("MINIST", draw_callback, &image);
    std::fill(minist_->input_image_.begin(), minist_->input_image_.end(), 0.0f);
    while(true) {
        cv::imshow("MINIST", image);
        if(cv::waitKey(1) == 27) {
            break;
        }
        cv::Mat image_scaled;
        cv::resize(image, image_scaled, cv::Size(MINIST::width_, MINIST::height_));
        for(int i=0; i!=MINIST::height_; ++i) {
            for(int j=0; j!=MINIST::width_; ++j) {
                minist_->input_image_[i * MINIST::width_ + j] = image_scaled.at<uchar>(i, j) / 255.0f;
            }
        }
        minist_->Run();
        std::cout << "The number is: " << minist_->result_ << std::endl;
    }
    return 0;
}
