import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)
# a @ b is equivalent to np.matmul(a, b)
c_mm_relu = np.maximum(a_np @ b_np, 0)
# Numpy底层调用OpenBLAS等底层库以及自己的一些C语言实现来执行计算

# 等效实现
def lnumpy_m_relu(A:np.ndarray,B:np.ndarray,C:np.ndarray):
    Y = np.empty(A.shape,dtype='float32')
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if k == 0:
                    Y[i,j] = 0
                Y[i,j] += A[i,k] * B[k,j]
            C[i,j] = max(Y[i,j],0)

c_np = np.empty((128, 128), dtype=dtype)
lnumpy_m_relu(a_np, b_np, c_np)
np.testing.assert_allclose(c_np, c_mm_relu, rtol=1e-5)

# 使用TVM的TensorIR来实现
# TVM的TensorIR是一种用于表示计算的中间表示，它是一种低级的表示，用于表示计算的数据流和计算的依赖关系
# 它是由TVMScript语言来实现的，这是一种嵌入在Python AST中的DSL，用于表示TVM的中间表示
@tvm.script.ir_module # 修饰器，用于声明MyModule是一个TVM的IRModule，IRModule是在机器学习编译中保存张量函数集合的容器。
class MyModule:
    @T.prim_func # 修饰器，用于声明一个TVM的primitive function
    def mm_relu(
        A:T.Buffer((128, 128), dtype="float32"),
        B:T.Buffer((128, 128), dtype="float32"),
        C:T.Buffer((128, 128), dtype="float32")):
        T.func_attr({"global_symbol": "mm_relu","tir.noalias": True})
        # T.func_attr是一个语法糖，用来声明函数的属性，global_symbol对应函数名，tir.noalias表示所有的缓冲存储器都是不重叠的。
        Y = T.alloc_buffer((128, 128), "float32")
        for i,j,k in T.grid(128,128,128): # T.grid()返回的是一个生成器，可以用for循环来遍历,是TensorIR的一个语法糖
            with T.block("Y"): # T.block()是一个语法糖，用来定义一个block,这个block可以包含多个axis以及围绕这些axis的计算
                vi = T.axis.spatial(128, i) # 定义了一个空间轴，表示i的范围是[0,128)，声明了轴的属性spatial，表示这是一个空间轴
                vj = T.axis.spatial(128, j) 
                vk = T.axis.reduce(128, k) # 还有一种属性是reduce，表示这是一个规约轴
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i,j in T.grid(128,128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))

# 块设计有助于我们进行机器学习编译分析，我们总是在空间轴上做并行化，但是在规约轴上做并行化需要考虑数据依赖关系

# 每个块轴直接映射到外部循环迭代器的情况下，我们可以使用 T.axis.remap 在一行中声明所有块轴。
@tvm.script.ir_module
class MyModuleWithAxisRemapSugar:
    @T.prim_func
    def mm_relu(
        A:T.Buffer((128, 128), dtype="float32"),
        B:T.Buffer((128, 128), dtype="float32"),
        C:T.Buffer((128, 128), dtype="float32")):
        T.func_attr({"global_symbol": "mm_relu","tir.noalias": True})
        Y = T.alloc_buffer((128, 128), "float32")
        for i,j,k in T.grid(128,128,128):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

        for i,j in T.grid(128,128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


# IRModule可以包含多个张量函数
@tvm.script.ir_module
class MyModuleWithTwoFunctions:
    @T.prim_func
    def mm(A:T.Buffer((128, 128), dtype="float32"),
            B:T.Buffer((128, 128), dtype="float32"),
            Y:T.Buffer((128, 128), dtype="float32")):
        T.func_attr({"global_symbol": "mm","tir.noalias": True})
        for i,j,k in T.grid(128,128,128):
            with T.block("Y"):
                vi,vj,vk = T.axis.remap("SSR",[i,j,k])
                with T.init():
                    Y[vi,vj] = T.float32(0)
                Y[vi,vj] = Y[vi,vj] + A[vi,vk] * B[vk,vj]
        
        
    @T.prim_func
    def relu(A: T.Buffer((128, 128), "float32"),
             B: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "relu", "tir.noalias": True})
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], T.float32(0))

# 张量函数的变换
# 张量函数的变换是指对张量函数的变换，包括对块轴的变换，对块的变换，对循环迭代器的变换等

def lnumpy_mm_relu_v2(A:np.ndarray,B:np.ndarray,C:np.ndarray):
    Y = np.empty(A.shape,dtype='float32')
    for i in range(128):
        for j0 in range(32):
            for k in range(128):
                for j1 in range(4):
                    j = j0*4+j1
                    if k == 0:
                        Y[i,j] = 0
                    Y[i,j] += A[i,k] * B[k,j]
    for i in range(128):
        for j in range(128):
            C[i,j] = max(Y[i,j],0)

c_np_v2 = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu_v2(a_np, b_np, c_np_v2)
np.testing.assert_allclose(c_np_v2, c_mm_relu, rtol=1e-5)

# 进行张量函数变换
sch = tvm.tir.Schedule(MyModule)
# 创建一个辅助的Schedule对象，用于对张量函数进行变换

block_Y = sch.get_block("Y", func_name="mm_relu")
i,j,k = sch.get_loops(block_Y)
# 获得块和循环的引用

j0,j1 = sch.split(j, factors=[None,4])
# 将j轴分裂为j0和j1两个轴，j0的范围是[0,32)，j1的范围是[0,4)

sch.reorder(i,j0,k,j1)
# 重新排列块轴的顺序

block_C = sch.get_block("C", func_name="mm_relu")
sch.reverse_compute_at(block_C, j0)
# 将C块的计算位置移动到j0轴

sch.decompose_reduction(block_Y, k)
# 将Y的元素初始化和计算分解开

#最后变换后的实现等价于lnumpy_mm_relu_v3
def lnumpy_mm_relu_v3(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j0 in range(32):
            # Y_init
            for j1 in range(4):
                j = j0 * 4 + j1
                Y[i, j] = 0
            # Y_update
            for k in range(128):
                for j1 in range(4):
                    j = j0 * 4 + j1
                    Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
            # C
            for j1 in range(4):
                j = j0 * 4 + j1
                C[i, j] = max(Y[i, j], 0)

c_np = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu_v3(a_np, b_np, c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol=1e-5)