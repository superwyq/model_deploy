import numpy as np
import onnx
import onnxruntime as ort
import torch


ort_session = ort.InferenceSession("/home/wyq/hobby/model_deploy/onnx/export_onnx/super_resolution.onnx",
                                   providers=['CUDAExecutionProvider','CPUExecutionProvider'])
# onnxruntime中加载模型为InferenceSession对象，可以指定使用的provider，如CUDAExecutionProvider和CPUExecutionProvider
# providers是一个列表，可以指定多个provider，onnxruntime会根据模型的输入和输出自动选择合适的provider
# 通过onnxruntime的InferenceSession对象，可以进行模型的推理
#还可以通过options参数指定模型的一些配置，如logid等
print(ort.get_available_providers()) # 获取onnxruntime支持的provider
nparray = np.random.randn(1, 1, 224, 224).astype(np.float32)
ortvalue = ort.OrtValue.ortvalue_from_numpy(nparray)
# onnxruntime中的OrtValue对象可以将numpy数组转换为onnxruntime的tensor对象,ortValue对象可以作为模型的输入
#onnxruntime默认将numpy数组转换为CPU上的tensor对象，可以通过指定设备名称将numpy数组转换为指定设备上的tensor对象
print("ortvalue device:",ortvalue.device_name())  # 获取ortvalue对象所在的设备名称
print("ortvalue data type:",ortvalue.data_type())  # 获取ortvalue对象的数据类型
print("ortvalue shape:",ortvalue.shape())  # 获取ortvalue对象的形状
print("if ortvalue is tensor:",ortvalue.is_tensor())  # 判断ortvalue对象是否是tensor对象
np.array_equal(ortvalue.numpy(), nparray)  # 判断ortvalue对象的数据是否与numpy数组相等
result = ort_session.run(None, {'input': ortvalue})
# onnxruntime的InferenceSession对象的run方法可以进行模型推理，输入参数为模型的输入，返回值为模型的输出
# run方法的第一个参数为模型的输出名称，第二个参数为模型的输入，返回值为模型的输出

#onnxruntime的InferenceSession对象还提供了io_binding方法，可以将模型的输入和输出绑定到指定的tensor对象上
X_ortvalue = ort.OrtValue.ortvalue_from_numpy(nparray,'cuda',0)
io_binding = ort_session.io_binding() # 创建io_binding对象
io_binding.bind_input('input',
                      device_type=X_ortvalue.device_name(),
                      device_id=0,
                      element_type=np.float32,
                      shape=X_ortvalue.shape(),
                      buffer_ptr=X_ortvalue.data_ptr()) # 绑定模型的输入
# buffer_ptr参数为tensor对象的数据指针，可以通过tensor对象的data_ptr方法获取
io_binding.bind_output('output') # 绑定模型的输出
#onnxruntime可以为output动态分配内存，也可以通过bind_output方法指定output的形状和数据类型
ort_session.run_with_iobinding(io_binding) # 使用io_binding方法进行模型推理

Y = io_binding.copy_outputs_to_cpu()[0] # 将模型的输出从GPU上拷贝到CPU上
print("Y shape:", Y.shape)

#onnxruntime io_binding方法可以提高模型推理的效率，特别是在模型的输入和输出形状不变的情况下
#onnxruntime io_binding还可以绑定到pytorch的tensor对象上，可以通过pytorch的tensor对象的data_ptr方法获取数据指针
Y_tensor = torch.zeros(1, 1, 672, 672).cuda()
io_binding.bind_output(
    'output',
    device_type=Y_tensor.device.type,
    device_id=Y_tensor.device.index,
    element_type=np.float32,
    shape=Y_tensor.shape,
    buffer_ptr=Y_tensor.data_ptr())
ort_session.run_with_iobinding(io_binding)
print("Y_tensor shape:", Y_tensor.shape)