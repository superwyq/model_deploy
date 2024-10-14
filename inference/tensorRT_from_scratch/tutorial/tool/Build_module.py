import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit



# 创建logger
# 创建builder
# 创建builder_config
# 创建network，通过onnx模型或者手动添加层,如果有不支持的层，需要自定义plugin
# 序列化engine
# 反序列化engine
# 创建执行器executor
# 执行推理，将输入拷贝到显存，执行推理，将输出拷贝到内存
# 释放资源 

logger = trt.Logger(trt.Logger.INFO)
# 日志器,等级分为：VERBOSE, INFO, WARNING, ERROR, INTERNAL_ERROR
# VERBOSE和INFO最常用，VERBOSE会输出模型优化的详细信息，INFO只会输出模型的输入输出信息
builder = trt.Builder(logger)
# 网络构建器，仅作为构建网络的工具，不用于设置网络属性
profile = builder.create_optimization_profile()
profile.set_shape("input", (1, 1, 28, 28), (1, 1, 28, 28), (1, 1, 28, 28))
builder_conf = builder.create_builder_config()
# 构建器配置，用于设置构建器的属性，可以设置的有：最大显存，int8量化的校正器，设置推理精度等
builder_conf.add_optimization_profile(profile)

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#网络本体，可以设置网络的输入输出，添加层，设置层的属性等
inputTensor = network.add_input(name="input", dtype=trt.float32, shape=(1,1, 28, 28))
identityLayer = network.add_identity(inputTensor)
network.mark_output(identityLayer.get_output(0))
# 添加输入层，添加一个identity层，将输入直接输出，作为输出层

serializedNetwork = builder.build_serialized_network(network, builder_conf)
# 序列化网络，将网络序列化为字节流，可以保存到文件中，也可以用于反序列化

# # 保存到网络文件
# with open("test.engine", "wb") as f:
#     f.write(serializedNetwork)

engine = trt.Runtime(logger).deserialize_cuda_engine(serializedNetwork)
# 反序列化网络，将序列化的网络反序列化为可执行的引擎，可以用于推理
lTensorName = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
print(lTensorName)
# 获取引擎的输入输出名称

context = engine.create_execution_context()
# 创建执行上下文，用于执行推理

context.set_input_shape(lTensorName[0], (1, 1, 28, 28))
# 设置输入形状

hInput = np.random.random((1, 1, 28, 28)).astype(np.float32)
# 创建输入数据,host端
dInput = cuda.mem_alloc(hInput.nbytes)
# 分配显存
houtput = np.empty((1, 1, 28, 28), dtype=np.float32)
# 创建输出数据
doutput = cuda.mem_alloc(houtput.nbytes)
# 分配显存
context.set_tensor_address(lTensorName[0], int(dInput))
context.set_tensor_address(lTensorName[1], int(doutput))
# 设置输入输出显存地址

#复制数据从host到device
cuda.memcpy_htod(dInput, hInput)
# 执行推理
context.execute_async_v3(0)
# 复制数据从device到host
cuda.memcpy_dtoh(houtput, doutput)
print(houtput)
# 释放显存

