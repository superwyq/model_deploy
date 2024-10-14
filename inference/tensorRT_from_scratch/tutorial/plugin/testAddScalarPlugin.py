#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import ctypes
import os

import numpy as np
import tensorrt as trt
from cuda import cudart

soFile = "./AddScalarPlugin.so"
np.set_printoptions(precision=3, linewidth=200, suppress=True)
# np.set_printoptions函数的作用是设置打印时的精度、行宽、是否使用科学计数法等。其中的三个
# 参数含义分别是：precision：设置浮点数的精度，即小数点后的位数；linewidth：设置输出的行宽；suppress：当suppress=True时，表示不输出小数点后面的数字，即将小数部分四舍五入
np.random.seed(31193)
cudart.cudaDeviceSynchronize()

def printArrayInformation(x, info="", n=5):
    if 0 in x.shape:
        print('%s:%s' % (info, str(x.shape)))
        return
    x = x.astype(np.float32)
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])
    return

def check(a, b, weak=False, checkEpsilon=1e-5, info=""):
    if a.shape != b.shape:
        print("Error shape: A%s : B%s" % (str(a.shape), str(b.shape)))
        return
    if weak:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    maxAbsDiff = np.max(np.abs(a - b))
    meanAbsDiff = np.mean(np.abs(a - b))
    maxRelDiff = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    meanRelDiff = np.mean(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    res = "%s:%s,MaxAbsDiff=%.2e,MeanAbsDiff=%.2e,MaxRelDiff=%.2e,MeanRelDiff=%.2e," % (info, res, maxAbsDiff, meanAbsDiff, maxRelDiff, meanRelDiff)
    index = np.argmax(np.abs(a - b))
    valueA, valueB= a.flatten()[index], b.flatten()[index]
    shape = a.shape
    indexD = []
    for i in range(len(shape) - 1, -1, -1):
        x = index % shape[i]
        indexD = [x] + indexD
        index = index // shape[i]
    res += "WorstPair=(%f:%f)at%s" %(valueA, valueB, str(indexD))
    print(res)
    return

def addScalarCPU(inputH, scalar):
    return [inputH[0] + scalar]

def getAddScalarPlugin(scalar):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "AddScalar":
            parameterList = []
            parameterList.append(trt.PluginField("scalar", np.float32(scalar), trt.PluginFieldType.FLOAT32))
            # PluginField类的作用是定义插件的属性，其中的三个参数分别是属性的名称、属性的值和属性的数据类型。
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
            # create_plugin函数的作用是创建一个插件，其中的两个参数分别是插件的名称和插件的属性集合。
    return None

def run(shape, scalar):
    testCase = "<shape=%s,scalar=%f>" % (shape, scalar)
    trtFile = "./model-Dim%s.plan" % str(len(shape))
    print("Test %s" % testCase)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    # trt.init_libnvinfer_plugins函数的作用是初始化TensorRT库中的插件，其中的两个参数分别是日志级别和插件库的路径。
    ctypes.cdll.LoadLibrary(soFile)
    # ctypes.cdll.LoadLibrary函数的作用是加载指定的动态链接库，其中的参数是动态链接库的路径。
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()

        inputT0 = network.add_input("inputT0", trt.float32, [-1 for i in shape])
        profile.set_shape(inputT0.name, [1 for i in shape], [8 for i in shape], [32 for i in shape])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2([inputT0], getAddScalarPlugin(scalar))
        # add_plugin_v2函数的作用是向网络中添加一个插件层，其中的两个参数分别是输入张量列表和插件。
        network.mark_output(pluginLayer.get_output(0))
        # mark_output函数的作用是标记网络的输出张量，其中的参数是张量。
        engineString = builder.build_serialized_network(network, config)
        # build_serialized_network函数的作用是构建序列化的网络，其中的两个参数分别是网络和配置。
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
        # deserialize_cuda_engine函数的作用是反序列化一个CUDA引擎，其中的参数是序列化的引擎。

    nIO = engine.num_io_tensors
    # num_io_tensors属性的作用是获取引擎的输入输出张量的数量。
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    # get_tensor_name函数的作用是获取引擎的输入输出张量的名称。
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
    # get_tensor_mode函数的作用是获取引擎的输入输出张量的模式，其中的参数是张量的名称。
    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], shape)
    #for i in range(nIO):
    #    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    bufferH.append(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    # np.arange函数的作用是创建一个等差数组，其中的参数是数组的大小。np.prod函数的作用是计算数组的元素个数。
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
        # 初始化一个空数组，数组的形状是引擎的输入输出张量的形状，数组的数据类型是引擎的输出张量的数据类型。
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
        # cudart.cudaMalloc函数的作用是在GPU上分配一块内存，其中的参数是内存的大小。
        # 为推理输入输出张量分配内存。

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        # cudart.cudaMemcpy函数的作用是在GPU之间复制内存，其中的四个参数分别是目标内存、源内存、内存的大小和复制的方向。
        # 将模型的输入张量从CPU复制到GPU。

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))
        # set_tensor_address函数的作用是设置张量的地址，其中的两个参数分别是张量的名称和地址。

    context.execute_async_v3(0)
    # execute_async_v3函数的作用是异步执行推理，其中的参数是批次大小。

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        # 将模型的输出张量从GPU复制到CPU。

    outputCPU = addScalarCPU(bufferH[:nInput], scalar)
    """
    for i in range(nInput):
        printArrayInformation(bufferH[i])
    for i in range(nInput, nIO):
        printArrayInformation(bufferH[i])
    for i in range(nInput, nIO):
        printArrayInformation(outputCPU[i - nInput])
    """
    check(bufferH[nInput:][0], outputCPU[0], True)

    for b in bufferD:
        cudart.cudaFree(b)
        # 释放GPU上的内存。
    print("Test %s finish!\n" % testCase)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")

    run([32], 1)
    run([32, 32], 1)
    run([16, 16, 16], 1)
    run([8, 8, 8, 8], 1)
    run([32], 1)
    run([32, 32], 1)
    run([16, 16, 16], 1)
    run([8, 8, 8, 8], 1)

    print("Test all finish!")