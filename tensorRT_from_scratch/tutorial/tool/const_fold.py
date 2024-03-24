import onnx
import numpy as np
import onnx_graphsurgeon as gs
from collections import OrderedDict

input = gs.Variable(name="input", dtype=np.float32, shape=(1, 1, 28, 28))
conv_output = gs.Variable(name="conv_output", dtype=np.float32, shape=None)
bn_output = gs.Variable(name="bn_output", dtype=np.float32, shape=None)
relu_output = gs.Variable(name="relu_output", dtype=np.float32, shape=None)
add1_output = gs.Variable(name="add1_output", dtype=np.float32, shape=None)
constant1 = gs.Constant(name="constant1", values=np.array([1], dtype=np.float32))
add2_output = gs.Variable(name="add2_output", dtype=np.float32, shape=None)
constant2 = gs.Constant(name="constant2", values=np.array([2], dtype=np.float32))
add3_output = gs.Variable(name="add3_output", dtype=np.float32, shape=None)
constant4 = gs.Constant(name="constant4", values=np.array([4], dtype=np.float32))
output = gs.Variable(name="output", dtype=np.float32, shape=None)

nodelist = []

conv_node = gs.Node(op="Conv", inputs=[input], outputs=[conv_output])
conv_node.attrs = OrderedDict([["kernel_shape", [5, 5]], ["pads", [2, 2, 2, 2]]])
nodelist.append(conv_node)
bn_node = gs.Node(op="BatchNormalization", inputs=[conv_output], outputs=[bn_output])
nodelist.append(bn_node)
relu_node = gs.Node(op="Relu", inputs=[bn_output], outputs=[relu_output])
nodelist.append(relu_node)
add1_node = gs.Node(op="Add", inputs=[relu_output, constant1], outputs=[add1_output])
nodelist.append(add1_node)
add2_node = gs.Node(op="Add", inputs=[constant1, constant2], outputs=[add2_output])
nodelist.append(add2_node)
add3_node = gs.Node(op="Add", inputs=[add2_output, add1_output], outputs=[add3_output])
nodelist.append(add3_node)
add4_node = gs.Node(op="Add", inputs=[add3_output, constant4], outputs=[output])
nodelist.append(add4_node)

graph = gs.Graph(nodes=nodelist, inputs=[input], outputs=[output])
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "no_const_fold.onnx")
onnx.save(gs.export_onnx(graph.fold_constants().cleanup()), "const_fold.onnx")
