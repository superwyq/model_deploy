import torch.onnx
import torch
from pre_data import *
from models.yolo import myYOLO

input_height = 416
input_width = 416
input_size = [input_height, input_width]
device = torch.device("cpu")
model = myYOLO(device, input_size=input_size, num_classes=VOC_CLASSES_NUM, trainable=False)
model.load_state_dict(torch.load("/home/wyq/hobby/model_deploy/tensorRT_from_scratch/YOLOv1_PyTorch/weights/voc/yolo_64.4_68.5_71.5.pth"))


input = torch.randn(1, 3, input_height, input_width, requires_grad=True)
torch_out = model(input)

torch.onnx.export(model, input, "yolov1.onnx", export_params=True,
                    opset_version=11, do_constant_folding=True,
                    input_names = ['input'], output_names = ['output'],
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})