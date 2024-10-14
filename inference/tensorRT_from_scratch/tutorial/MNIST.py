import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit
import cv2 as cv

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
parser = trt.OnnxParser(network, logger)

onnxFile = "/home/wyq/hobby/model_deploy/onnx/onnxruntime/MNIST/mnist.onnx"
res = parser.parse_from_file(onnxFile)

shape = (1, 1, 28, 28)
output_shape = (1, 10)

inputTensor = network.get_input(0)
profile.set_shape(inputTensor.name, shape, shape, shape)
config.add_optimization_profile(profile)

serializedNetwork = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(serializedNetwork)
context = engine.create_execution_context()

hInput = np.random.random(shape).astype(np.float32)
# hInput = cv.imread("/home/wyq/hobby/model_deploy/source/number_0.png")
dInput = cuda.mem_alloc(hInput.nbytes)
hOutput = np.empty(output_shape, dtype=np.float32)
dOutput = cuda.mem_alloc(hOutput.nbytes)

lTensorName = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
context.set_input_shape(lTensorName[0], (1, 1, 28, 28))
context.set_tensor_address(lTensorName[0], dInput)
context.set_tensor_address(lTensorName[1], dOutput)

cuda.memcpy_htod(dInput, hInput)
context.execute_async_v3(0)
cuda.memcpy_dtoh(hOutput, dOutput)
print(hOutput)
