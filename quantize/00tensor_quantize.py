import torch
from torch import nn
import torch.nn.functional as F

# Create a model
class Model_demo(nn.Module):
    def __init__(self):
        super(Model_demo, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# An instance of your model.
model = Model_demo()
quantized_model = torch.quantization.quantize_dynamic(model=model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

print(quantized_model.fc1.weight())
print(model.fc1.weight)
# 可以看到量化后的参数是有一些误差的