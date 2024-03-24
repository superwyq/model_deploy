import torch as tr
from torchvision import models,datasets,transforms as T

from PIL import Image
import numpy as np
import os
from torchvision.models.resnet import ResNet50_Weights

resnet50 = models.resnet50(weights = ResNet50_Weights.DEFAULT)

os.system("curl -o imagenet_classes.txt https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

image_height = 224
image_width = 224

x = tr.randn(1, 3, image_height, image_width, requires_grad=True)
torch_out = resnet50(x)
tr.onnx.export(resnet50, x, "resnet50.onnx", export_params=True,
                opset_version=12, do_constant_folding=True,
                input_names = ['input'], output_names = ['output'],
                dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                              'output' : {0 : 'batch_size'}})


