import onnx
import numpy as np
# Given a bool scalar input cond.
# return constant tensor x if cond is True, otherwise return constant tensor y.

cond = onnx.helper.make_tensor_value_info( # 创建输入
    "cond", onnx.TensorProto.BOOL, []
)

then_out = onnx.helper.make_tensor_value_info( # 创建then输出
    "then_out", onnx.TensorProto.FLOAT, [5]
)
else_out = onnx.helper.make_tensor_value_info( # 创建else输出
    "else_out", onnx.TensorProto.FLOAT, [5]
)

x = np.array([1, 2, 3, 4, 5]).astype(np.float32) # 创建then输出的值
y = np.array([5, 4, 3, 2, 1]).astype(np.float32) # 创建else输出的值

then_const_node = onnx.helper.make_node( # 创建then输出的节点
    "Constant",
    inputs=[],
    outputs=["then_out"],
    value=onnx.numpy_helper.from_array(x),
)

else_const_node = onnx.helper.make_node( # 创建else输出的节点
    "Constant",
    inputs=[],
    outputs=["else_out"],
    value=onnx.numpy_helper.from_array(y),
)

then_body = onnx.helper.make_graph( # 创建then的子图
    [then_const_node], "then_body", [], [then_out]
)

else_body = onnx.helper.make_graph( # 创建else的子图
    [else_const_node], "else_body", [], [else_out]
)

if_node = onnx.helper.make_node( # 创建if节点
    "If",
    inputs=["cond"],
    outputs=["res"],
    then_branch=then_body,
    else_branch=else_body,
)

res = onnx.helper.make_tensor_value_info("res", onnx.TensorProto.FLOAT, [5]) # 创建输出，这个输出是if节点的输出
graph = onnx.helper.make_graph( # 创建主图
    [if_node], "test_if", [cond], [res]
)
onnx.save_model( # 保存模型
    onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 11)]),
    "if.onnx",
)