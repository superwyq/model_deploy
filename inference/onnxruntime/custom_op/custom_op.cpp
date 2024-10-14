#include<onnxruntime/onnxruntime_cxx_api.h>
#include<onnxruntime/onnxruntime_lite_custom_op.h>
using namespace Ort::Custom;

void KernelOne(const Tensor<float>& X,
               const Tensor<float>& Y,
               Tensor<float>& Z) {
  auto input_shape = X.Shape();
  auto x_raw = X.Data();
  auto y_raw = Y.Data();
  auto z_raw = Z.Allocate(input_shape);
  for (int64_t i = 0; i < Z.NumberOfElement(); ++i) {
    z_raw[i] = x_raw[i] + y_raw[i];
  }
}

int main() {
  Ort::CustomOpDomain v1_domain{"v1"};
  // please make sure that custom_op_one has the same lifetime as the consuming session
  std::unique_ptr<OrtLiteCustomOp> custom_op_one{Ort::Custom::CreateLiteCustomOp("CustomOpOne", "CPUExecutionProvider", KernelOne)};
  v1_domain.Add(custom_op_one.get());
  Ort::SessionOptions session_options;
  session_options.Add(v1_domain);
  // create a session with the session_options ...
}
