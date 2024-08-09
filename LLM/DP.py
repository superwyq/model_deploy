import torch as tr
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
from torch.nn import DataParallel

model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
print(tr.__version__)

device = tr.device('cuda:1,2' if tr.cuda.is_available() else 'cpu')
model = DataParallel(model, device_ids=[1, 2])
model = model.to(device)
