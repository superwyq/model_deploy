import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2 as cv
N_CLASSES = 80 # yolov5 class label number
BATCH_SIZE=1
PRECISION= np.float32


dummy_input_batch = np.zeros((BATCH_SIZE,3,640,640),dtype=PRECISION)

f = open("yolov5s.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

output = np.empty(N_CLASSES, dtype = PRECISION) # Need to set both input and output precisions to FP16 to fully enable FP16

d_input = cuda.mem_alloc(1 * dummy_input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()


def predict(batch): # result gets copied into output
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # Execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # Syncronize threads
    stream.synchronize()
    return output

pred = predict(dummy_input_batch)
print(pred.shape)