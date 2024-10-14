import numpy as np
import onnx
import onnx_graphsurgeon as gs

input_tensor = gs.Variable(name="input", dtype=np.float32, shape=(1, 1, 28, 28))
conv_output = gs.Variable(name="conv_output", dtype=np.float32, shape=None)
id_output = gs.Variable(name="id_output", dtype=np.float32, shape=None)
output_tensor = gs.Variable(name="output", dtype=np.float32, shape=None) #shape为None表示任意形状

conv_node = gs.Node(op="Conv", inputs=[input_tensor], outputs=[conv_output])
identity_node = gs.Node(op="Identity",name="goal_node", inputs=[conv_output], outputs=[id_output])
identity_node2 = gs.Node(op="Identity", inputs=[id_output], outputs=[output_tensor])

graph = gs.Graph(nodes=[conv_node, identity_node,identity_node2], inputs=[input_tensor], outputs=[output_tensor])
graph.cleanup().toposort() #清理图中无用的节点和tensor，拓扑排序
onnx.save(gs.export_onnx(graph), "test.onnx") #将图导出为onnx模型
# graph = gs.import_onnx(onnx.load("test.onnx")) #将onnx模型导入为图
for node in graph.nodes:
    print(node)
    if node.op == "Identity" and node.name == "goal_node":
        constant0 = gs.Constant(name="constant0", values=np.array([0], dtype=np.float32))
        add_output = gs.Variable(name="add_output", dtype=np.float32, shape=None)
        add_node = gs.Node(op="Add", inputs=[node.outputs[0], constant0], outputs=[add_output])

        graph.nodes.append(add_node)
        index = node.o().inputs.index(node.outputs[0]) #找到当前节点的输出在下一个节点的输入中的索引
        node.o().inputs[index] = add_output #将下一个节点的输入替换为新的tensor
graph.cleanup().toposort()

print("#"*20, "After change", "#"*20)
for node in graph.nodes:
    print(node)
onnx.save(gs.export_onnx(graph), "test1.onnx") #将图导出为onnx模型

# 删除节点
for node in graph.nodes:
    print(node)
    if node.op == "Identity" and node.name == "goal_node":
        id_input_ = node.inputs[0]
        id_output_ = node.outputs[0]
        for sub_node in graph.nodes:
            if id_input_ in sub_node.inputs:
                sub_node.inputs[sub_node.inputs.index(id_input_)] = id_output_
graph.cleanup().toposort() #清理图中无用的节点会自动把没用的节点删除

# 替换节点,也可以先插入节点再删除要替换的节点，一样的效果
graph = gs.import_onnx(onnx.load("test.onnx")) #将onnx模型导入为图

for node in graph.nodes:
    if node.op == "Identity" and node.name == "goal_node":
        node.op = "Add"
        node.name = "add_node"
        node.inputs = [node.inputs[0], gs.Constant(name="constant0", values=np.array([0], dtype=np.float32))]

graph.cleanup().toposort()