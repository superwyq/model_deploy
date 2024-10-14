# Given an input sequence [x1, ..., xN], sum up its elements using a scan
# returning the final state (x1+x2+...+xN) as well the scan_output
# [x1, x1+x2, ..., x1+x2+...+xN]
#
# create graph to represent scan body
import numpy as np
import onnx

sum_in = onnx.helper.make_tensor_value_info(
    "sum_in", onnx.TensorProto.FLOAT, [2]
)
next = onnx.helper.make_tensor_value_info("next", onnx.TensorProto.FLOAT, [2])
sum_out = onnx.helper.make_tensor_value_info(
    "sum_out", onnx.TensorProto.FLOAT, [2]
)
scan_out = onnx.helper.make_tensor_value_info(
    "scan_out", onnx.TensorProto.FLOAT, [2]
)
add_node = onnx.helper.make_node(
    "Add", inputs=["sum_in", "next"], outputs=["sum_out"]
)
id_node = onnx.helper.make_node(
    "Identity", inputs=["sum_out"], outputs=["scan_out"]
)
scan_body = onnx.helper.make_graph(
    [add_node, id_node], "scan_body", [sum_in, next], [sum_out, scan_out]
)
# create scan op node
node = onnx.helper.make_node(
    "Scan",
    inputs=["initial", "x"],
    outputs=["y", "z"],
    num_scan_inputs=1,
    body=scan_body,
)
# create inputs for sequence-length 3, inner dimension 2
initial = np.array([0, 0]).astype(np.float32).reshape((2,))
x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32).reshape((3, 2))
# final state computed = [1 + 3 + 5, 2 + 4 + 6]
y = np.array([9, 12]).astype(np.float32).reshape((2,))
# scan-output computed
z = np.array([1, 2, 4, 6, 9, 12]).astype(np.float32).reshape((3, 2))

# create graph
initial_info = onnx.helper.make_tensor_value_info(
    "initial", onnx.TensorProto.FLOAT, initial.shape
)
x_info = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, x.shape)
y_info = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, y.shape)
z_info = onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, z.shape)
graph = onnx.helper.make_graph(
    [node], "test_scan", [initial_info, x_info], [y_info, z_info]
)
# create model
model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 11)])
# save model
onnx.save_model(model, "scan.onnx")

#inference
import onnxruntime as rt
sess = rt.InferenceSession("scan.onnx")
res = sess.run(None, {"initial": initial, "x": x})
print(res)
