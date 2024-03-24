/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../include/cookbookHelper.cuh"

namespace
{
static const char *PLUGIN_NAME {"AddScalar"};
static const char *PLUGIN_VERSION {"1"};
} // namespace

namespace nvinfer1
{
class AddScalarPlugin : public IPluginV2DynamicExt // 定义AddScalarPlugin类，继承IPluginV2DynamicExt类
{
private:
    const std::string name_; //算子名称
    std::string       namespace_; //算子所属的域
    struct
    {
        float scalar;
    } m_;

public:
    AddScalarPlugin() = delete; //禁止默认构造函数
    AddScalarPlugin(const std::string &name, float scalar);
    AddScalarPlugin(const std::string &name, const void *buffer, size_t length); //构造函数
    ~AddScalarPlugin();

    // Method inherited from IPluginV2
    const char *getPluginType() const noexcept override; //获取插件类型，noexcept表示该函数不会抛出异常，override表示该函数是虚函数
    const char *getPluginVersion() const noexcept override; //获取插件版本
    int32_t     getNbOutputs() const noexcept override; //获取输出张量的数量
    int32_t     initialize() noexcept override; //初始化插件
    void        terminate() noexcept override; //终止插件，释放资源
    size_t      getSerializationSize() const noexcept override; //获取序列化后的大小
    void        serialize(void *buffer) const noexcept override; //序列化
    void        destroy() noexcept override; //销毁插件，当context或engine被销毁时，插件也会被销毁
    void        setPluginNamespace(const char *pluginNamespace) noexcept override; //设置插件的命名空间
    const char *getPluginNamespace() const noexcept override; //获取插件的命名空间
    //当我们的模型来自onnx的时候，命名空间，版本等信息会被保存在onnx模型中，这个函数就是用来获取这些信息的
    //一般不用我们自己设置，而是由onnx模型中的信息来设置
    //如果这些信息设置不对，会导致onnxparser解析模型的时候出错，无法识别插件

    // Method inherited from IPluginV2Ext
    DataType getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept override;
    void     attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept override;
    void     detachFromContext() noexcept override;

    // Method inherited from IPluginV2DynamicExt
    IPluginV2DynamicExt *clone() const noexcept override;
    DimsExprs            getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;
    // getOutputDimensions，向TensorRT报告输出张量的形状，outputIndex是指输出张量的索引
    bool                 supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    // supportsFormatCombination，检查输入和输出张量的格式是否支持，pos是指输入张量的索引，inOut是指输入和输出张量的描述符, nbInputs是指输入张量的数量，nbOutputs是指输出张量的数量
    // 尽量多的支持格式组合，以便TensorRT可以选择最佳的格式组合
    void                 configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override;
    // configurePlugin，配置插件，in是指输入张量的描述符，nbInputs是指输入张量的数量，out是指输出张量的描述符，nbOutputs是指输出张量的数量
    // 在推理期前调用该函数，用于将插件中的动态维度转换为静态维度
    size_t               getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override;
    // getWorkspaceSize，获取插件所需的工作空间大小，inputs是指输入张量的描述符，nbInputs是指输入张量的数量，outputs是指输出张量的描述符，nbOutputs是指输出张量的数量
    // 在推理期前调用该函数，用于计算插件所需的工作空间大小，向TensorRT报告工作空间的大小
    int32_t              enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
    // enqueue，执行插件的推理，inputDesc是指输入张量的描述符，outputDesc是指输出张量的描述符，inputs是指输入张量的数据，outputs是指输出张量的数据，workspace是指工作空间，stream是指CUDA流
    // 在推理期间调用该函数，用于执行插件的推理。不要在enqueue中调用cudaMalloc或cudaFree等CUDA API，会造成性能下降
    // 原因我猜是因为前面getworkspaceSize已经分配了空间，如果这里再进行分配，会使之前针对内存分配做的优化失效
protected:
    // To prevent compiler warnings，使用using声明，将基类的成员函数引入到子类中，避免编译器警告
    using nvinfer1::IPluginV2::enqueue;
    using nvinfer1::IPluginV2::getOutputDimensions;
    using nvinfer1::IPluginV2::getWorkspaceSize;
    using nvinfer1::IPluginV2Ext::configurePlugin;
};

class AddScalarPluginCreator : public IPluginCreator 
// 定义一个AddScalarPluginCreator类，继承于IPluginCreator，PluginCreator是一个工厂类，用于创建Plugin
{
private:
    static PluginFieldCollection    fc_;
    static std::vector<PluginField> attr_;
    std::string                     namespace_;

public:
    AddScalarPluginCreator();
    ~AddScalarPluginCreator();
    const char                  *getPluginName() const noexcept override;
    const char                  *getPluginVersion() const noexcept override;
    const PluginFieldCollection *getFieldNames() noexcept override;
    IPluginV2DynamicExt         *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override;
    // 接受一个插件名称和插件属性集合，返回一个新的插件实例
    IPluginV2DynamicExt         *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override;
    // 接受一个插件名称和序列化数据，返回一个新的插件实例
    void                         setPluginNamespace(const char *pluginNamespace) noexcept override;
    // 设置插件的命名空间
    const char                  *getPluginNamespace() const noexcept override;
    // 获取插件的命名空间
};

} // namespace nvinfer1