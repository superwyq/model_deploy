import deepspeed as dp
import torch as tr
import torchvision.models as models
import time as t

model = models.mobilenet_v2(pretrained=True)

begin = t.time()
model()

model_engine = dp.init_inference(model=model)
model_engine.eval()
model_engine = model_engine.cuda()

input_tensor = tr.randn(1, 3, 224, 224,dtype=tr.half)
input_tensor = input_tensor.cuda()
result = model_engine(input_tensor)
print(result.shape)
print(result.argmax()) 