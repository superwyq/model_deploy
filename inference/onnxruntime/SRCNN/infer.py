import onnxruntime as ort
import numpy as np
import cv2 as cv


options = ort.SessionOptions()
options.enable_profiling = True

ort_session = ort.InferenceSession("/home/wyq/hobby/model_deploy/onnx/export_onnx/super_resolution.onnx",
                                   sess_options=options,
                                   providers=['CUDAExecutionProvider','CPUExecutionProvider'])
#加载模型，InferenceSession是一个类，用于加载模型
#providers参数是一个list，指定使用的设备，这里指定了CUDA和CPU，按照顺序优先使用CUDA，如果CUDA不可用则使用CPU
#可以通过ort.get_available_providers()查看可用的provider
#还可以通过options参数设置一些参数，比如enable_profiling，enable_mem_pattern，logid等

input_img = cv.imread("/home/wyq/hobby/model_deploy/cuda_from_scratch/test.jpg",cv.IMREAD_GRAYSCALE).astype(np.float32)
#opencv读取图像存储type是uint8，所以需要转换成float32
input_img = cv.resize(input_img, (224, 224))
input_img = np.expand_dims(input_img, axis=0)
#gray读取的是单通道图像，所以需要扩展维度，扩展到CHW
input_img = np.expand_dims(input_img, axis=0)
#扩展到NCHW
print(input_img.shape)
ort_inputs = {'input': input_img}

ort_outputs = ort_session.run(['output'], ort_inputs)[0]


ort_outpts = np.squeeze(ort_outputs, axis=0)
ort_outpts = np.clip(ort_outpts, 0, 255)
ort_outpts = np.transpose(ort_outpts, (1, 2, 0)).astype(np.uint8)

cv.imshow("output", ort_outpts)
cv.waitKey(0)
