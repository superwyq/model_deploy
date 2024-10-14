import torch

# define a floating point model where some layers could be statically quantized
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub 将float tensor转化为量化表示
        self.quant = torch.ao.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub 将量化表示转化为float tensor
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        # 自己手动指定量化模型中的量化点
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        # 自己决定何时将量化表示转化为float tensor
        x = self.dequant(x)
        return x


model_fp32 = M()

# 模型必须设置为eval模式，以便在量化过程中，模型的行为和量化后的行为一致
model_fp32.eval()

# 模型量化配置，里面包括了默认的量化配置，可以通过`torch.ao.quantization.get_default_qconfig('x86')`获取
# 对于PC端的量化，推荐使用`x86`，对于移动端的量化，推荐使用`qnnpack`
# 其他的量化配置，比如选择对称量化还是非对称量化，以及MinMax还是L2Norm校准技术，都可以在这里指定
model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')

# 手动进行融合，将一些常见的操作融合在一起，以便后续的量化
# 常见的融合包括`conv + relu`和`conv + batchnorm + relu`
model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'relu']])


# 准备模型，插入观察者，观察激活张量，观察者用于校准量化参数
model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)

# 进行校准，这里输入需要使用代表性的数据，以便观察者能够观察到激活张量的分布，从而计算出量化参数
input_fp32 = torch.randn(4, 1, 4, 4)
model_fp32_prepared(input_fp32)


# 将模型转化为量化模型，这里会将权重量化，计算并存储每个激活张量的scale和bias值，以及用量化实现替换关键操作
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

# 运行量化模型，这里的计算都是在int8上进行的
res = model_int8(input_fp32)