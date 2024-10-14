#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include<iomanip>

using namespace std;
using namespace cv;
using namespace Ort;

struct Net_config
{
	float confThreshold; // 置信度阈值，小于阈值认为该框中物体不是这个class
	float nmsThreshold;  // NMS非极大值抑制阈值
	float objThreshold;  // 物体检测阈值，小于该阈值认为框中没有物体
	string modelpath;   //模型文件地址
};

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

int endsWith(string s, string sub) {
	return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
}

const float anchors_640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
								 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
								 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };

class YOLO
{
public:
	YOLO(Net_config config);
	Mat detect(Mat& frame);

private:
	float* anchors;   //anchor框,yolo中预置了640分辨率的anchor尺寸，每两个数表示一个anchor的size，例如(10,13)，yolo中有三个尺度的输出，每个尺度的anchor数为3
	int num_stride; // stride的数量，yolo中有三个尺度的输出，每个尺度的stride为8,16,32
	int inpWidth; //输入宽度
	int inpHeight; //输入高度
	int nout; //输出的每个proposal的维度，一共85个值，分别是xmin,ymin,xmax,ymax,box_score,然后是类别的score，一共80个类别
	int num_proposal; //输出的proposal数量
	vector<string> class_names; //类别名称
	int num_class; //类别数量
	int seg_num_class; //分割类别数量，用不到

	float confThreshold; // 置信度阈值，小于阈值认为该框中物体不是这个class
	float nmsThreshold; // NMS非极大值抑制阈值
	float objThreshold; // 物体检测阈值，小于该阈值认为框中没有物体
	const bool keep_ratio = true; //是否保持原图比例
	vector<float> input_image_; //输入图像
	void normalize_(Mat img); //归一化
	void nms(vector<BoxInfo>& input_boxes); //非极大值抑制
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left); //图像缩放

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5-7"); //初始化环境
	Ort::Session *ort_session = nullptr; //模型session
	SessionOptions sessionOptions = SessionOptions(); //模型session配置
	vector<string> input_names; //输入节点名称
	vector<string> output_names; //输出节点名称
	vector<vector<int64_t>> input_node_dims; // 输入节点维度
	vector<vector<int64_t>> output_node_dims; // 输出节点维度
};

YOLO::YOLO(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;

	string classesFile = "/home/wyq/hobby/model_deploy/onnx/onnxruntime/YoloV5/class.names"; //类别名称文件
	string model_path = config.modelpath;
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	std::vector<std::string> avaliable_providers = GetAvailableProviders();
	auto cuda_provider = std::find(avaliable_providers.begin(), avaliable_providers.end(), "CUDA");
	if(cuda_provider != avaliable_providers.end())
	{
		cout<<"cuda provider is available"<<endl;
		OrtCUDAProviderOptions cuda_options = OrtCUDAProviderOptions{}; //使用cuda推理
		sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
	}
	else
	{
		cout<<"cuda provider is not available"<<endl;
	}

	ort_session = new Session(env, model_path.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	cout<<numInputNodes<<endl;
	cout<<numOutputNodes<<endl;
	for (int i = 0; i < numInputNodes; i++) //获取输入节点信息
	{
		auto name = ort_session->GetInputNameAllocated(i, allocator);
		input_names.push_back(string(name.get()));
		cout<<input_names[i]<<endl;
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++) //获取输出节点信息
	{
		auto name = ort_session->GetOutputNameAllocated(i, allocator);
		output_names.push_back(string(name.get()));
		cout<<output_names[i]<<endl;
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->nout = output_node_dims[0][2];
	this->num_proposal = output_node_dims[0][1];

	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();
	this->anchors = (float*)anchors_640;
	this->num_stride = 3; //设置stride数量
}

Mat YOLO::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLO::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;

			}
		}
	}
}


void YOLO::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); //按照score降序排列
	vector<float> vArea(input_boxes.size()); //存储每个box的面积
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false); //存储每个box是否被抑制
	for (int i = 0; i < int(input_boxes.size()); ++i) //遍历所有box
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j) //计算当前box与其它box的IOU
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nmsThreshold)
			{
				isSuppressed[j] = true; //抑制IOU大于阈值的box，也就是这个box和box[i]重叠度很高
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
	//这里用到了C++11中的新特性lambda，匿名函数，可以自己去了解一下，推荐深入理解C++11：C++11新特性解析与应用这本书，对于C++11讲解的很好。
}

Mat YOLO::detect(Mat& frame)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	//vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());
	const array<const char*,1> input_names_array = { input_names[0].c_str() };
	const array<const char*,1> output_names_array = { output_names[0].c_str()};
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, input_names_array.data(), &input_tensor_, 1, output_names_array.data(), output_names_array.size());
	//输出的组成：每个proposal由5个部分组成，分别是xmin,ymin,xmax,ymax,box_score,然后是类别的score，一共80个类别，所以一共85个值，

	/////generate proposals
	vector<BoxInfo> generate_boxes; //存储所有的box
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww; //计算原图和resize后图像的比例，用于将box坐标映射到原图
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	for (int n = 0; n < this->num_stride; n++)   
	{
		const float stride = pow(2, n + 3); //计算stride步长，不同的尺度对应不同的stride
		int num_grid_x = (int)ceil((this->inpWidth / stride)); //计算x方向的网格数量
		int num_grid_y = (int)ceil((this->inpHeight / stride)); //计算y方向的网格数量
		for (int q = 0; q < 3; q++)    ///anchor，每个尺度有三个anchor
		{
			const float anchor_w = this->anchors[n * 6 + q * 2]; //计算anchor的宽度
			const float anchor_h = this->anchors[n * 6 + q * 2 + 1]; //计算anchor的高度
			for (int i = 0; i < num_grid_y; i++) //遍历y方向的网格
			{
				for (int j = 0; j < num_grid_x; j++) //遍历x方向的网格
				{
					float box_score = pdata[4]; //输出的第四个值是box的置信度
					if (box_score > this->objThreshold) //如果置信度大于阈值，才认为检测到了物体
					{
						int max_ind = 0;
						float max_class_socre = 0;
						for (int k = 0; k < num_class; k++) //遍历80个类别，找到最大的类别得分
						{
							if (pdata[k + 5] > max_class_socre)
							{
								max_class_socre = pdata[k + 5];
								max_ind = k;
							}
						}
						max_class_socre *= box_score; //类别得分乘以box的置信度，得到最终的得分
						if (max_class_socre > this->confThreshold) //如果最终得分大于阈值，才认为检测到了物体，还原box坐标到原图
						{ 
							float cx = (pdata[0] * 2.f - 0.5f + j) * stride;  ///cx
							float cy = (pdata[1] * 2.f - 0.5f + i) * stride;   ///cy
							float w = powf(pdata[2] * 2.f, 2.f) * anchor_w;   ///w
							float h = powf(pdata[3] * 2.f, 2.f) * anchor_h;  ///h

							float xmin = (cx - padw - 0.5 * w)*ratiow;
							float ymin = (cy - padh - 0.5 * h)*ratioh;
							float xmax = (cx - padw + 0.5 * w)*ratiow;
							float ymax = (cy - padh + 0.5 * h)*ratioh;

							generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, max_ind });
						}
					}
					pdata += nout; //移动到下一个proposal
				}
			}
		}
		
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms(generate_boxes);

	for (size_t i = 0; i < generate_boxes.size(); ++i) //画框
	{
		int xmin = int(generate_boxes[i].x1);
		int ymin = int(generate_boxes[i].y1);
		rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
		string label = format("%.2f", generate_boxes[i].score);
		label = this->class_names[generate_boxes[i].label] + ":" + label;
		putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
	return frame; //返回画好框的图像,其实不用返回也可以，因为是引用传递
}


int main()
{
	Net_config yolo_nets = { 0.3, 0.5, 0.3,"/home/wyq/hobby/model_deploy/onnx/onnxruntime/YoloV5/build/yolov5s.onnx" };


	YOLO yolo_model(yolo_nets);
	Mat srcimg;
	// VideoCapture cap("/home/wyq/hobby/model_deploy/video.mp4");
	VideoCapture cap=VideoCapture(0);
	cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
	cap.set(CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CAP_PROP_FRAME_HEIGHT, 480);
	cap.set(CAP_PROP_FPS, 60);

	while(true)
	{
		double inference_time = 0;
		double fps = 0.0;
		cap >> srcimg;
		if(srcimg.empty())
		{
			cout<<"can not load image"<<endl;
			break;
		}
		double begin = static_cast<double>(getTickCount());
		yolo_model.detect(srcimg);
		
		inference_time = (static_cast<double>(getTickCount()) - begin) / getTickFrequency();
		cout<<"inference time:"<<inference_time<<endl;
		fps = 1.0f / inference_time;
		putText(srcimg, "FPS:"+to_string(fps), Point(16, 32),FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 0, 255));
		imshow("yolo", srcimg);

		cout<<"fps:"<<fps<<endl;
		if(waitKey(1) == 27)
			break;
	}
}